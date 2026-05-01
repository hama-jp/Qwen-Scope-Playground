<script lang="ts">
  import { createNote, fetchFeatureSamples } from '../api';
  import {
    corpusStatus,
    featureStack,
    inspectorTarget,
    notes
  } from '../stores';
  import type { FeatureSample } from '../types';

  let layerInput = $state(10);
  let idxInput = $state(16503);
  let samples = $state<FeatureSample[]>([]);
  let loading = $state(false);
  let error = $state<string | null>(null);
  let lastFetchedKey = $state<string | null>(null);

  // Sync inputs when an external target is set (e.g., chip click).
  $effect(() => {
    const t = $inspectorTarget;
    if (!t) return;
    layerInput = t.layer;
    idxInput = t.feature_idx;
    void load(t.layer, t.feature_idx);
  });

  async function load(l: number, i: number) {
    const key = `${l}:${i}`;
    if (key === lastFetchedKey && !error) return;
    loading = true;
    error = null;
    try {
      const r = await fetchFeatureSamples(l, i, 10);
      samples = r.samples;
      lastFetchedKey = key;
    } catch (e) {
      const msg = (e as Error).message;
      error = msg === 'corpus_unavailable' ? 'corpus_unavailable' : msg;
      samples = [];
    } finally {
      loading = false;
    }
  }

  function onLoad() {
    void load(layerInput, idxInput);
  }

  function addToStack() {
    const idx = idxInput;
    featureStack.update((stack) => {
      if (stack.some((f) => f.idx === idx)) return stack;
      return [...stack, { idx, strength_multiplier: 1.0 }];
    });
  }

  let noteLabel = $state('');
  let noteMemo = $state('');
  let savingNote = $state(false);
  async function saveNote() {
    if (!noteLabel.trim()) return;
    savingNote = true;
    try {
      const created = await createNote({
        layer: layerInput,
        feature_idx: idxInput,
        label: noteLabel.trim(),
        memo: noteMemo
      });
      notes.update((ns) => [created, ...ns]);
      noteLabel = '';
      noteMemo = '';
    } catch (e) {
      error = (e as Error).message;
    } finally {
      savingNote = false;
    }
  }
</script>

<aside class="rounded-lg border border-neutral-800 bg-neutral-900/40 p-4 space-y-4 text-sm">
  <h2 class="text-sm font-semibold text-neutral-300 flex items-center justify-between">
    <span>Inspector</span>
    {#if $corpusStatus && !$corpusStatus.available}
      <span class="text-xs px-1.5 py-0.5 rounded border border-amber-700 text-amber-400">
        corpus not built
      </span>
    {:else if $corpusStatus?.available}
      <span class="text-xs text-neutral-500"
        >{$corpusStatus.num_samples} samples · top-{$corpusStatus.top_n}</span
      >
    {/if}
  </h2>

  <div class="flex items-end gap-2">
    <div class="space-y-1">
      <label for="ins-layer" class="text-xs text-neutral-400">Layer</label>
      <input
        id="ins-layer"
        type="number"
        min="0"
        max="27"
        bind:value={layerInput}
        class="w-16 bg-neutral-900 border border-neutral-700 rounded px-2 py-1 text-sm font-mono"
      />
    </div>
    <div class="space-y-1 flex-1">
      <label for="ins-idx" class="text-xs text-neutral-400">Feature idx</label>
      <input
        id="ins-idx"
        type="number"
        min="0"
        bind:value={idxInput}
        class="w-full bg-neutral-900 border border-neutral-700 rounded px-2 py-1 text-sm font-mono"
      />
    </div>
    <button
      type="button"
      class="px-3 py-1 rounded bg-neutral-800 hover:bg-neutral-700 border border-neutral-700 text-xs"
      onclick={onLoad}
      disabled={loading}
    >
      {loading ? '…' : 'Load'}
    </button>
  </div>

  <div class="flex gap-2">
    <button
      type="button"
      class="text-xs px-2 py-1 rounded bg-emerald-800 hover:bg-emerald-700 text-white"
      onclick={addToStack}>+ stack</button
    >
  </div>

  {#if error === 'corpus_unavailable' || ($corpusStatus && !$corpusStatus.available)}
    <div class="text-xs text-neutral-400 leading-relaxed border border-dashed border-neutral-700 rounded p-3">
      Corpus not yet mined. Run:
      <pre class="mt-1 bg-neutral-900 p-2 rounded font-mono text-[11px] text-amber-300">python -m backend.corpus_mine --layer 10 --num-samples 1000</pre>
      and reload the page to see top-activating snippets here.
    </div>
  {:else if error}
    <div class="text-xs text-red-400 break-words">{error}</div>
  {/if}

  {#if samples.length > 0}
    <div class="space-y-2 max-h-[60vh] overflow-y-auto pr-1">
      {#each samples as s (s.rank)}
        <div class="rounded border border-neutral-800 bg-neutral-950 p-2 space-y-1">
          <div class="flex items-center justify-between text-[11px] font-mono text-neutral-500">
            <span>#{s.rank} · sample {s.sample_id} · pos {s.highlight_pos}</span>
            <span class="text-amber-300">act {s.activation.toFixed(1)}</span>
          </div>
          <div class="text-[12px] font-mono leading-relaxed whitespace-pre-wrap break-words">
            {#each s.tokens.slice(s.context_start, s.context_end) as tok, i}
              {@const absPos = s.context_start + i}
              <span class:bg-amber-700={absPos === s.highlight_pos} class:text-white={absPos === s.highlight_pos}
                >{tok}</span
              >
            {/each}
          </div>
        </div>
      {/each}
    </div>
  {:else if !loading && !error}
    <div class="text-xs text-neutral-500 italic">
      Enter a feature index above and Load to see its top-activating snippets.
    </div>
  {/if}

  <details class="border-t border-neutral-800 pt-3">
    <summary class="text-xs text-neutral-400 cursor-pointer hover:text-neutral-200"
      >+ Save note</summary
    >
    <div class="mt-2 space-y-2">
      <input
        type="text"
        placeholder="Label (e.g. Python keyword)"
        bind:value={noteLabel}
        class="w-full bg-neutral-900 border border-neutral-700 rounded px-2 py-1 text-sm"
      />
      <textarea
        placeholder="Memo (optional)"
        bind:value={noteMemo}
        class="w-full h-16 bg-neutral-900 border border-neutral-700 rounded px-2 py-1 text-xs"
      ></textarea>
      <button
        type="button"
        class="w-full text-xs px-2 py-1 rounded bg-emerald-700 hover:bg-emerald-600 disabled:opacity-50"
        disabled={savingNote || !noteLabel.trim()}
        onclick={saveNote}
      >
        {savingNote ? 'saving…' : 'Save note'}
      </button>
    </div>
  </details>
</aside>
