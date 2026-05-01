<script lang="ts">
  import type { TokenWithLogits } from '../types';

  type Props = {
    tokens: TokenWithLogits[];
    running: boolean;
    // 'baseline' (variant 0) renders in green, every other variant in
    // amber. Picking hue by category keeps baseline vs steered visually
    // distinct even at a glance — the lightness still encodes confidence.
    variantKind: 'baseline' | 'steered';
  };
  let { tokens, running, variantKind }: Props = $props();

  const HUE = $derived(variantKind === 'baseline' ? 140 : 30);

  function bgFor(p: number | undefined): string {
    if (p == null) return 'transparent';
    // Lightness 100% = transparent, 35% = full saturation. A linear map is
    // good enough — readers care about high-confidence vs low-confidence,
    // not exact percentile.
    const clamped = Math.max(0, Math.min(1, p));
    const L = 100 - clamped * 65;
    return `hsl(${HUE} 70% ${L}% / 0.55)`;
  }

  function fmt(p: number): string {
    return p.toFixed(3);
  }

  // Floating tooltip — single instance per Heatmap, positioned on hover via
  // viewport coordinates. We can't use position:absolute inside the token
  // span any more because ResultCard's body is `overflow-y:auto`, which
  // clips both axes (browsers don't honour mixed overflow-x:visible +
  // overflow-y:auto). position:fixed escapes the clip box entirely.
  let hovered = $state<number | null>(null);
  let pos = $state<{ left: number; top: number; flipUp: boolean }>({
    left: 0,
    top: 0,
    flipUp: false,
  });

  const TT_MAX_W = 320;
  const TT_GAP = 6;

  function placeFor(target: HTMLElement) {
    const r = target.getBoundingClientRect();
    let left = r.left;
    if (left + TT_MAX_W > window.innerWidth - 8) {
      left = Math.max(8, window.innerWidth - TT_MAX_W - 8);
    }
    // Flip to above the token if there isn't room below (rough estimate;
    // the table is ~150px tall with k=5, plus padding).
    const TT_EST_H = 160;
    const flipUp = r.bottom + TT_GAP + TT_EST_H > window.innerHeight - 8;
    const top = flipUp ? r.top - TT_GAP : r.bottom + TT_GAP;
    pos = { left, top, flipUp };
  }

  function onEnter(e: MouseEvent, i: number) {
    placeFor(e.currentTarget as HTMLElement);
    hovered = i;
  }
  function onLeave() {
    hovered = null;
  }

  let hoveredTok = $derived(hovered != null ? tokens[hovered] : null);
  let chosenInTopk = $derived.by(() => {
    if (!hoveredTok || !hoveredTok.topk) return true;
    return hoveredTok.topk.some(([s]) => s === hoveredTok.text);
  });
</script>

<div class="font-mono text-sm leading-relaxed whitespace-pre-wrap break-words">
  {#each tokens as tok, i (i)}
    {@const cp = tok.chosen_prob}
    {@const lowConf = cp != null && cp < 0.05}
    {@const highConf = cp != null && cp >= 0.95}
    <span
      class="heat-token"
      class:font-bold={highConf}
      class:low-conf={lowConf}
      style:background-color={bgFor(cp)}
      onmouseenter={(e) => onEnter(e, i)}
      onmouseleave={onLeave}
    >{tok.text}</span>{/each}{#if running}<span class="text-neutral-500">▍</span>{/if}
</div>

{#if hoveredTok && hoveredTok.topk}
  <div
    class="heat-tt"
    class:flip-up={pos.flipUp}
    style:left="{pos.left}px"
    style:top="{pos.top}px"
  >
    <table class="text-[11px]">
      <thead>
        <tr class="text-neutral-400">
          <th class="px-1 text-left">#</th>
          <th class="px-1 text-left">token</th>
          <th class="px-1 text-right">prob</th>
        </tr>
      </thead>
      <tbody>
        {#each hoveredTok.topk as [s, p], r (r)}
          <tr class:chosen={hoveredTok.token_id != null && s === hoveredTok.text}>
            <td class="px-1 text-neutral-500">{r}</td>
            <td class="px-1 font-mono">{JSON.stringify(s)}</td>
            <td class="px-1 text-right tabular-nums">{fmt(p)}</td>
          </tr>
        {/each}
      </tbody>
    </table>
    {#if hoveredTok.chosen_prob != null && !chosenInTopk}
      <div class="px-1 pt-1 text-[11px] text-amber-400">
        chosen token outside top-{hoveredTok.topk.length} (rep_penalty?)
      </div>
    {/if}
  </div>
{/if}

<style>
  .heat-token {
    border-radius: 2px;
    padding: 1px 0;
  }
  .heat-token.low-conf {
    /* Without an outline, chosen_prob ≈ 0 cells become invisible because
       the background is nearly transparent — the dotted outline says
       "something was emitted here, hover to see what". */
    outline: 1px dotted rgba(255, 255, 255, 0.2);
    outline-offset: -1px;
  }
  .heat-tt {
    position: fixed;
    z-index: 50;
    padding: 4px;
    background: rgb(20 20 22);
    border: 1px solid rgb(64 64 64);
    border-radius: 4px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
    pointer-events: none;
    max-width: 320px;
  }
  .heat-tt.flip-up {
    /* When we flipped to above the token, top is the token's top edge —
       translate up by our own height so the bottom edge lands at top - gap. */
    transform: translateY(-100%);
  }
  :global(.heat-tt tr.chosen) {
    background: rgba(250, 204, 21, 0.15);
  }
</style>
