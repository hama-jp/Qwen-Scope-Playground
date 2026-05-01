<script lang="ts">
  import { generating, heatmapMode, variants } from '../stores';
  import ResultCard from './ResultCard.svelte';

  let baseline = $derived(
    $variants.find((v) => v.strength === 0) ?? $variants[0] ?? null
  );
</script>

<section class="rounded-lg border border-neutral-800 bg-neutral-900/40 p-4 space-y-3">
  <div class="flex items-center justify-between">
    <h2 class="text-sm font-semibold text-neutral-300">Results</h2>
    <label
      class="text-xs text-neutral-400 inline-flex items-center gap-1.5 cursor-pointer select-none"
      title="When ON, Generate captures top-k token probabilities so each cell shows model confidence."
    >
      <input
        type="checkbox"
        bind:checked={$heatmapMode}
        disabled={$generating}
        class="accent-emerald-600"
      />
      Heatmap (token confidence)
    </label>
  </div>
  {#if $variants.length === 0}
    <div class="text-sm text-neutral-500 italic">
      no runs yet — set up the compose pane and click <span class="font-mono">Generate</span>.
    </div>
  {:else}
    <div class="flex gap-3 overflow-x-auto pb-2">
      {#each $variants as v (v.variant_id)}
        <ResultCard variant={v} {baseline} mode={$heatmapMode ? 'heatmap' : 'plain'} />
      {/each}
    </div>
  {/if}
</section>
