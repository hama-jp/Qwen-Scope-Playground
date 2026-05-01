<script lang="ts">
  import type { VariantState } from '../types';
  import TokenHeatmap from './TokenHeatmap.svelte';

  type Props = {
    variant: VariantState;
    baseline?: VariantState | null;
    mode?: 'plain' | 'heatmap';
  };
  let { variant, baseline = null, mode = 'plain' }: Props = $props();

  // Effective mode: heatmap requires per-token logits to actually have arrived.
  // If the toggle is on but this variant has no logits (e.g. a card from a
  // previous v0.2 run still in state), fall back to plain rather than render
  // unstyled spans.
  let effectiveMode = $derived(mode === 'heatmap' && variant.has_logits ? 'heatmap' : 'plain');

  // Plain-mode pieces: position-level diff against baseline. Each piece is
  // whatever the streamer emitted in one chunk (usually a token-with-leading-
  // space). The diff highlight is suppressed in heatmap mode — color encodes
  // confidence there, not divergence.
  let pieces = $derived.by<{ text: string; changed: boolean }[]>(() => {
    if (!baseline || baseline.variant_id === variant.variant_id) {
      return variant.tokens.map((t) => ({ text: t.text, changed: false }));
    }
    return variant.tokens.map((t, i) => ({
      text: t.text,
      changed: baseline.tokens[i]?.text !== t.text
    }));
  });

  // Average chosen_prob over all tokens that carry logits. Skipped when there
  // are none, so the badge only appears in heatmap mode.
  let avgConf = $derived.by(() => {
    const probs = variant.tokens
      .map((t) => t.chosen_prob)
      .filter((p): p is number => typeof p === 'number');
    if (probs.length === 0) return null;
    return probs.reduce((a, b) => a + b, 0) / probs.length;
  });

  function copy() {
    navigator.clipboard.writeText(variant.text).catch(() => {});
  }
</script>

<div
  class="flex flex-col rounded border border-neutral-800 bg-neutral-900/60 w-72 shrink-0 max-h-[60vh]"
>
  <header class="flex items-center justify-between px-3 py-2 border-b border-neutral-800">
    <div>
      <div class="text-xs text-neutral-500">variant {variant.variant_id}</div>
      <div class="text-sm font-mono">
        {variant.label}
      </div>
    </div>
    <div class="flex items-center gap-2">
      {#if effectiveMode === 'heatmap' && avgConf != null}
        <span
          class="text-[10px] px-1.5 py-0.5 rounded bg-neutral-800 border border-neutral-700 font-mono tabular-nums"
          title="average chosen_prob across emitted tokens"
        >
          μ {avgConf.toFixed(2)}
        </span>
      {/if}
      {#if !variant.done}
        <span class="text-xs text-amber-400 animate-pulse">●</span>
      {:else}
        <span class="text-xs text-emerald-400">●</span>
      {/if}
      <button
        class="text-xs px-1.5 py-0.5 rounded bg-neutral-800 hover:bg-neutral-700 border border-neutral-700"
        onclick={copy}
        disabled={!variant.done}
        title="copy"
      >
        copy
      </button>
    </div>
  </header>

  <div class="flex-1 overflow-y-auto p-3 break-words">
    {#if effectiveMode === 'heatmap'}
      <TokenHeatmap
        tokens={variant.tokens}
        running={!variant.done}
        variantKind={variant.strength === 0 ? 'baseline' : 'steered'}
      />
    {:else}
      <div class="text-sm font-mono whitespace-pre-wrap">
        {#if mode === 'heatmap' && !variant.has_logits && variant.tokens.length > 0}
          <div
            class="text-[11px] text-amber-400 mb-2 italic"
            title="this variant was generated without with_topk_logits"
          >
            no heatmap data — re-Generate with the toggle on
          </div>
        {/if}
        {#each pieces as p, i (i)}
          <span class:diff-changed={p.changed}>{p.text}</span>
        {/each}{#if !variant.done}<span class="text-neutral-500">▍</span>{/if}
      </div>
    {/if}
  </div>
</div>
