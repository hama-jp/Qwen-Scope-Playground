<script lang="ts">
  import { onMount } from 'svelte';
  import ComposePane from '$lib/components/ComposePane.svelte';
  import ConceptSearch from '$lib/components/ConceptSearch.svelte';
  import Inspector from '$lib/components/Inspector.svelte';
  import NotesPane from '$lib/components/NotesPane.svelte';
  import ResultsGrid from '$lib/components/ResultsGrid.svelte';
  import { fetchCorpusStatus, fetchMeta, loadLayer, streamGenerate } from '$lib/api';
  import {
    corpusStatus,
    featureStack,
    generating,
    heatmapMode,
    lastError,
    layer,
    maxNewTokens,
    meta,
    prompt,
    repetitionPenalty,
    strengths,
    variants
  } from '$lib/stores';
  import type { GenerateRequest, SseEvent, VariantState } from '$lib/types';

  let abort: AbortController | null = null;

  onMount(async () => {
    try {
      $meta = await fetchMeta();
    } catch (e) {
      $lastError = `meta fetch failed: ${(e as Error).message}`;
    }
    try {
      $corpusStatus = await fetchCorpusStatus();
    } catch {
      $corpusStatus = { available: false };
    }
  });

  async function generate() {
    if ($generating) return;
    $lastError = null;

    // ensure layer is loaded
    if ($meta && !$meta.loaded_layers.includes($layer)) {
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
      } catch (e) {
        $lastError = (e as Error).message;
        return;
      }
    }

    // seed empty variant cards immediately so the UI renders all columns
    const seed: VariantState[] = $strengths.map((s, i) => ({
      variant_id: i,
      strength: s,
      label: s === 0 ? 'baseline' : `strength=${s}`,
      text: '',
      tokens: [],
      has_logits: false,
      done: false
    }));
    $variants = seed;
    $generating = true;

    abort = new AbortController();
    const req: GenerateRequest = {
      prompt: $prompt,
      max_new_tokens: $maxNewTokens,
      do_sample: false,
      repetition_penalty: $repetitionPenalty,
      layer: $layer,
      feature_stack: $featureStack,
      strengths: $strengths
    };
    if ($heatmapMode) {
      // Only send v0.3 fields when the toggle is on so the SSE bytes are
      // byte-identical to v0.2 in the default case (acceptance §6 row 1).
      req.with_topk_logits = true;
      req.topk_logits_k = 5;
    }

    try {
      await streamGenerate(req, handleEvent, abort.signal);
    } catch (e) {
      const msg = (e as Error).message ?? String(e);
      if (!msg.includes('aborted') && !msg.includes('AbortError')) {
        $lastError = msg;
      }
    } finally {
      $generating = false;
      // mark anything still streaming as done so the cursor stops
      variants.update((vs) => vs.map((v) => ({ ...v, done: true })));
      abort = null;
    }
  }

  function cancel() {
    abort?.abort();
  }

  function handleEvent(ev: SseEvent) {
    if (ev.event === 'variant_start') {
      const { variant_id, label } = ev.data;
      variants.update((vs) =>
        vs.map((v) => (v.variant_id === variant_id ? { ...v, label } : v))
      );
    } else if (ev.event === 'token') {
      const d = ev.data;
      // Narrow on `topk` presence: v0.2 payload is just {variant_id, token},
      // v0.3 carries token_id / chosen_prob / topk.
      const hasLogits = 'topk' in d && Array.isArray((d as { topk?: unknown }).topk);
      const piece = hasLogits
        ? {
            text: d.token,
            token_id: (d as { token_id: number }).token_id,
            chosen_prob: (d as { chosen_prob: number }).chosen_prob,
            topk: (d as { topk: [string, number][] }).topk
          }
        : { text: d.token };
      variants.update((vs) =>
        vs.map((v) =>
          v.variant_id === d.variant_id
            ? {
                ...v,
                tokens: [...v.tokens, piece],
                text: v.text + d.token,
                has_logits: v.has_logits || hasLogits
              }
            : v
        )
      );
    } else if (ev.event === 'variant_end') {
      const { variant_id, full_text } = ev.data;
      variants.update((vs) =>
        vs.map((v) =>
          v.variant_id === variant_id ? { ...v, done: true, text: full_text } : v
        )
      );
    } else if (ev.event === 'done') {
      // no-op; finally{} handles UI flag
    }
  }
</script>

<main class="p-4 space-y-4">
  <header class="flex items-baseline justify-between px-2">
    <h1 class="text-xl font-semibold tracking-tight">Qwen-Scope Playground</h1>
    {#if $meta}
      <div class="text-xs text-neutral-500 font-mono">
        {$meta.model_id} · {$meta.num_layers} layers · SAE width {$meta.sae_width} · top-{$meta.top_k}
      </div>
    {/if}
  </header>

  <!-- 3-pane: Discover | Compose+Results | Inspector
       Discover (left) hosts Concept search + Notes — both are sources of
       feature candidates that flow into the Compose Feature stack via the
       chip "+" buttons. Below xl (1280px) the Inspector collapses to a
       <details> so Compose stays usable on smaller screens. -->
  <div class="grid grid-cols-1 md:grid-cols-[22rem_minmax(0,1fr)] xl:grid-cols-[22rem_minmax(0,1fr)_22rem] gap-4 items-start">
    <div class="space-y-4 min-w-0">
      <ConceptSearch />
      <NotesPane />
    </div>

    <div class="space-y-4 min-w-0">
      <ComposePane onGenerate={generate} onCancel={cancel} />
      <ResultsGrid />
    </div>

    <div class="hidden xl:block">
      <Inspector />
    </div>
  </div>

  <details class="xl:hidden">
    <summary class="px-2 py-1 text-xs text-neutral-400 cursor-pointer hover:text-neutral-200">
      Open Inspector
    </summary>
    <div class="mt-2">
      <Inspector />
    </div>
  </details>
</main>
