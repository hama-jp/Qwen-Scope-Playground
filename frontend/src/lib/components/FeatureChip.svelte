<script lang="ts">
  import { featureStack, inspectorTarget, layer } from '../stores';

  type Props = {
    idx: number;
    label?: string | null;
    activation?: number | null;
    score?: number | null;
    title?: string;
  };
  let { idx, label = null, activation = null, score = null, title = '' }: Props = $props();

  function addToStack() {
    featureStack.update((stack) => {
      if (stack.some((f) => f.idx === idx)) return stack;
      return [...stack, { idx, strength_multiplier: 1.0 }];
    });
  }

  function openInspector() {
    inspectorTarget.set({ layer: $layer, feature_idx: idx });
  }
</script>

<span class="inline-flex items-stretch rounded border border-neutral-700 bg-neutral-900/70 text-xs font-mono overflow-hidden" {title}>
  <button
    type="button"
    class="px-2 py-0.5 hover:bg-neutral-800 text-emerald-300"
    onclick={openInspector}
    title="Open in Inspector"
  >
    #{idx}
  </button>
  {#if label}
    <span class="px-2 py-0.5 border-l border-neutral-700 text-neutral-300">{label}</span>
  {/if}
  {#if activation != null}
    <span class="px-2 py-0.5 border-l border-neutral-700 text-amber-300 tabular-nums">
      {activation.toFixed(1)}
    </span>
  {/if}
  {#if score != null}
    <span class="px-2 py-0.5 border-l border-neutral-700 text-amber-300 tabular-nums">
      {score >= 0 ? '+' : ''}{score.toFixed(1)}
    </span>
  {/if}
  <button
    type="button"
    class="px-2 py-0.5 border-l border-neutral-700 hover:bg-emerald-800 text-neutral-400 hover:text-white"
    onclick={addToStack}
    title="Add to stack"
  >
    +
  </button>
</span>
