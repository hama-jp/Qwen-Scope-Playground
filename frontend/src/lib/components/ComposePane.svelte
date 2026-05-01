<script lang="ts">
  import {
    featureStack,
    generating,
    lastError,
    layer,
    maxNewTokens,
    meta,
    prompt,
    promptEncode,
    promptEncodeLoading,
    repetitionPenalty,
    strengths
  } from '../stores';
  import { loadLayer, postEncode } from '../api';
  import FeatureChip from './FeatureChip.svelte';
  import FeatureStack from './FeatureStack.svelte';

  type Props = {
    onGenerate: () => void;
    onCancel: () => void;
  };
  let { onGenerate, onCancel }: Props = $props();

  let layerLoading = $state(false);
  let layerStatus = $derived.by(() => {
    if (!$meta) return '';
    return $meta.loaded_layers.includes($layer) ? 'loaded' : 'not loaded';
  });

  async function ensureLayer() {
    if (!$meta) return;
    if ($meta.loaded_layers.includes($layer)) return;
    layerLoading = true;
    try {
      await loadLayer($layer);
      meta.update((m) =>
        m
          ? {
              ...m,
              loaded_layers: [...new Set([...m.loaded_layers, $layer])].sort((a, b) => a - b)
            }
          : m
      );
      $lastError = null;
    } catch (e) {
      $lastError = (e as Error).message;
    } finally {
      layerLoading = false;
    }
  }

  // "Show top features for this prompt" — encode current prompt and surface chips.
  let skipFirst = $state(true);
  let showTopFeatures = $state(false);
  async function loadTopFeatures() {
    if (!$prompt.trim()) return;
    showTopFeatures = true;
    $promptEncodeLoading = true;
    try {
      const r = await postEncode({
        prompt: $prompt,
        layer: $layer,
        top_k_per_token: 8,
        skip_first: skipFirst
      });
      $promptEncode = r;
    } catch (e) {
      $lastError = (e as Error).message;
    } finally {
      $promptEncodeLoading = false;
    }
  }

  // Aggregate per-token top features into a single ranked chip list keyed by idx.
  // For each feature we track *one* peak — the (token, position) where its
  // activation was maximised across the prompt. Surfacing multiple "co-peak"
  // tokens used to confuse readers ("which one is the actual peak?"), and the
  // additional tokens were just side-effects of the per-token top-K cut.
  let aggregatedTopFeatures = $derived.by(() => {
    if (!$promptEncode)
      return [] as { idx: number; max_act: number; peak_token: string; peak_pos: number }[];
    const by_idx = new Map<number, { max_act: number; peak_token: string; peak_pos: number }>();
    for (let t = 0; t < $promptEncode.per_token.length; t++) {
      const tok = $promptEncode.tokens[t] ?? '';
      for (const [fi, act] of $promptEncode.per_token[t].feature_acts) {
        const cur = by_idx.get(fi);
        if (!cur || act > cur.max_act) {
          by_idx.set(fi, { max_act: act, peak_token: tok, peak_pos: t });
        }
      }
    }
    return [...by_idx.entries()]
      .map(([idx, v]) => ({ idx, ...v }))
      .sort((a, b) => b.max_act - a.max_act)
      .slice(0, 30);
  });

  // strengths editor: comma-separated
  let strengthsText = $state('0, 5, 20, 50, 100');
  $effect(() => {
    strengthsText = $strengths.join(', ');
  });
  function commitStrengths() {
    const parsed = strengthsText
      .split(/[\s,]+/)
      .filter((s) => s.length > 0)
      .map((s) => Number(s))
      .filter((n) => Number.isFinite(n));
    if (parsed.length > 0) {
      $strengths = parsed;
    }
  }
</script>

<section class="rounded-lg border border-neutral-800 bg-neutral-900/40 p-4 space-y-4">
  <h2 class="text-sm font-semibold text-neutral-300">Compose</h2>

  <div class="flex items-center gap-3">
    <label for="layer" class="text-sm text-neutral-400 w-16">Layer</label>
    <input
      id="layer"
      type="range"
      min="0"
      max={($meta?.num_layers ?? 28) - 1}
      bind:value={$layer}
      class="flex-1"
    />
    <span class="text-sm text-neutral-200 w-10 text-right tabular-nums">{$layer}</span>
    <span
      class="text-xs px-2 py-0.5 rounded border"
      class:border-emerald-700={layerStatus === 'loaded'}
      class:text-emerald-400={layerStatus === 'loaded'}
      class:border-neutral-700={layerStatus !== 'loaded'}
      class:text-neutral-400={layerStatus !== 'loaded'}
    >
      {layerStatus}
    </span>
    <button
      class="text-xs px-2 py-1 rounded bg-neutral-800 hover:bg-neutral-700 border border-neutral-700 disabled:opacity-50"
      onclick={ensureLayer}
      disabled={layerLoading || layerStatus === 'loaded'}
    >
      {layerLoading ? 'loading…' : 'Load layer'}
    </button>
  </div>

  <div class="space-y-1">
    <label for="prompt" class="text-sm text-neutral-400">Prompt</label>
    <textarea
      id="prompt"
      bind:value={$prompt}
      class="w-full h-24 bg-neutral-900 border border-neutral-700 rounded p-2 text-sm font-mono"
    ></textarea>
    <div class="flex items-center gap-3 pt-1">
      <button
        type="button"
        class="text-xs px-2 py-1 rounded bg-neutral-800 hover:bg-neutral-700 border border-neutral-700 disabled:opacity-50"
        onclick={loadTopFeatures}
        disabled={$promptEncodeLoading || !$prompt.trim()}
      >
        {$promptEncodeLoading ? 'encoding…' : 'Show top features for this prompt'}
      </button>
      <label class="text-xs text-neutral-400 inline-flex items-center gap-1">
        <input type="checkbox" bind:checked={skipFirst} class="accent-emerald-600" />
        skip first token
      </label>
    </div>
    {#if showTopFeatures && aggregatedTopFeatures.length > 0}
      <div class="flex flex-wrap gap-1.5 pt-2">
        {#each aggregatedTopFeatures as f (f.idx)}
          <FeatureChip
            idx={f.idx}
            activation={f.max_act}
            title={`peak: pos ${f.peak_pos} ${JSON.stringify(f.peak_token)}`}
          />
        {/each}
      </div>
    {/if}
  </div>

  <FeatureStack />

  <div class="flex flex-wrap gap-4 items-end">
    <div class="space-y-1 flex-1 min-w-[280px]">
      <label for="strengths" class="text-sm text-neutral-400"
        >Strength sweep (comma-separated)</label
      >
      <input
        id="strengths"
        type="text"
        class="w-full bg-neutral-900 border border-neutral-700 rounded px-2 py-1 text-sm font-mono"
        bind:value={strengthsText}
        onblur={commitStrengths}
        onkeydown={(e) => {
          if (e.key === 'Enter') {
            e.preventDefault();
            commitStrengths();
          }
        }}
      />
    </div>
    <div class="space-y-1">
      <label for="maxnew" class="text-sm text-neutral-400">max_new_tokens</label>
      <input
        id="maxnew"
        type="number"
        min="1"
        max="512"
        bind:value={$maxNewTokens}
        class="w-24 bg-neutral-900 border border-neutral-700 rounded px-2 py-1 text-sm"
      />
    </div>
    <div class="space-y-1">
      <label for="reppen" class="text-sm text-neutral-400">repetition_penalty</label>
      <input
        id="reppen"
        type="number"
        step="0.05"
        min="1"
        max="2"
        bind:value={$repetitionPenalty}
        class="w-24 bg-neutral-900 border border-neutral-700 rounded px-2 py-1 text-sm"
      />
    </div>
  </div>

  <div class="flex items-center gap-3">
    <button
      class="px-4 py-2 rounded bg-emerald-700 hover:bg-emerald-600 disabled:opacity-50 text-sm font-semibold"
      disabled={$generating || !$meta}
      onclick={() => {
        commitStrengths();
        onGenerate();
      }}
    >
      ▶ Generate
    </button>
    {#if $generating}
      <button
        class="px-3 py-2 rounded bg-neutral-800 hover:bg-red-900/60 border border-neutral-700 text-sm"
        onclick={onCancel}
      >
        ◼ Cancel
      </button>
    {/if}
    {#if $lastError}
      <span class="text-xs text-red-400">{$lastError}</span>
    {/if}
  </div>
</section>
