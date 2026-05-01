<script lang="ts">
  import { postFeatureSearch } from '../api';
  import { layer } from '../stores';
  import type { FeatureCandidate } from '../types';
  import FeatureChip from './FeatureChip.svelte';

  let positiveText = $state('def fibonacci(n):\nimport numpy as np\nclass Foo:');
  let negativeText = $state('Once upon a time\nI went to the store');
  let topK = $state(20);
  let loading = $state(false);
  let error = $state<string | null>(null);
  let results = $state<FeatureCandidate[]>([]);

  function splitSeeds(text: string): string[] {
    return text
      .split(/\n+/)
      .map((s) => s.trim())
      .filter((s) => s.length > 0);
  }

  async function search() {
    const pos = splitSeeds(positiveText);
    if (pos.length === 0) {
      error = 'positive seeds: at least one non-empty line required';
      return;
    }
    loading = true;
    error = null;
    try {
      const r = await postFeatureSearch({
        layer: $layer,
        seed_positive: pos,
        seed_negative: splitSeeds(negativeText),
        top_k: topK
      });
      results = r.candidates;
    } catch (e) {
      error = (e as Error).message;
      results = [];
    } finally {
      loading = false;
    }
  }
</script>

<details open class="rounded-lg border border-neutral-800 bg-neutral-900/40">
  <summary class="px-4 py-3 cursor-pointer text-sm font-semibold text-neutral-300 hover:text-white">
    Concept search
    {#if results.length > 0}
      <span class="text-neutral-500 font-normal text-xs">· {results.length} candidates</span>
    {/if}
  </summary>
  <div class="p-4 space-y-3 border-t border-neutral-800">
    <div class="space-y-3">
      <div class="space-y-1">
        <label for="cs-pos" class="text-xs text-neutral-400">Positive seeds (one per line)</label>
        <textarea
          id="cs-pos"
          bind:value={positiveText}
          class="w-full h-20 bg-neutral-900 border border-neutral-700 rounded px-2 py-1 text-xs font-mono"
        ></textarea>
      </div>
      <div class="space-y-1">
        <label for="cs-neg" class="text-xs text-neutral-400">Negative seeds (optional)</label>
        <textarea
          id="cs-neg"
          bind:value={negativeText}
          class="w-full h-20 bg-neutral-900 border border-neutral-700 rounded px-2 py-1 text-xs font-mono"
        ></textarea>
      </div>
    </div>
    <div class="flex items-center gap-3">
      <label for="cs-topk" class="text-xs text-neutral-400">top_k</label>
      <input
        id="cs-topk"
        type="number"
        min="1"
        max="100"
        bind:value={topK}
        class="w-20 bg-neutral-900 border border-neutral-700 rounded px-2 py-1 text-xs"
      />
      <button
        type="button"
        class="text-xs px-3 py-1 rounded bg-emerald-700 hover:bg-emerald-600 disabled:opacity-50"
        disabled={loading}
        onclick={search}
      >
        {loading ? 'searching…' : 'Search'}
      </button>
      {#if error}
        <span class="text-xs text-red-400">{error}</span>
      {/if}
    </div>
    {#if results.length > 0}
      <div class="flex flex-wrap gap-1.5">
        {#each results as c (c.feature_idx)}
          <FeatureChip
            idx={c.feature_idx}
            score={c.score}
            title={`pos=${c.pos_mean.toFixed(2)} neg=${c.neg_mean.toFixed(2)}`}
          />
        {/each}
      </div>
    {/if}
  </div>
</details>
