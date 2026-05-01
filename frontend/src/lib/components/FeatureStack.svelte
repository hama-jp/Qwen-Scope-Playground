<script lang="ts">
  import { featureStack } from '../stores';
  import type { FeatureRef } from '../types';

  function add() {
    featureStack.update((s) => [...s, { idx: 0, strength_multiplier: 1.0 }]);
  }
  function remove(i: number) {
    featureStack.update((s) => s.filter((_, k) => k !== i));
  }
  function patch(i: number, patch: Partial<FeatureRef>) {
    featureStack.update((s) => s.map((f, k) => (k === i ? { ...f, ...patch } : f)));
  }
</script>

<div class="space-y-2">
  <div class="flex items-center justify-between">
    <div class="text-sm text-neutral-400">Feature stack</div>
    <button
      class="text-xs px-2 py-1 rounded bg-neutral-800 hover:bg-neutral-700 border border-neutral-700"
      onclick={add}
    >
      + Add
    </button>
  </div>
  <div class="space-y-1">
    {#each $featureStack as f, i (i)}
      <div class="flex items-center gap-2">
        <span class="text-xs text-neutral-500 w-8">#{i}</span>
        <label class="text-xs text-neutral-500">idx</label>
        <input
          type="number"
          min="0"
          step="1"
          class="w-28 bg-neutral-900 border border-neutral-700 rounded px-2 py-1 text-sm"
          value={f.idx}
          oninput={(e) => patch(i, { idx: Number((e.target as HTMLInputElement).value) || 0 })}
        />
        <label class="text-xs text-neutral-500">×</label>
        <input
          type="number"
          step="0.1"
          class="w-24 bg-neutral-900 border border-neutral-700 rounded px-2 py-1 text-sm"
          value={f.strength_multiplier}
          oninput={(e) =>
            patch(i, {
              strength_multiplier: Number((e.target as HTMLInputElement).value) || 0
            })}
        />
        <button
          class="text-xs px-2 py-1 rounded bg-neutral-800 hover:bg-red-900/60 border border-neutral-700"
          onclick={() => remove(i)}
          aria-label="remove"
        >
          ×
        </button>
      </div>
    {/each}
    {#if $featureStack.length === 0}
      <div class="text-xs text-neutral-500 italic">no features — output will equal baseline</div>
    {/if}
  </div>
</div>
