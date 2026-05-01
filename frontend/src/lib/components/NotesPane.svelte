<script lang="ts">
  import { onMount } from 'svelte';
  import { createNote, deleteNote, listNotes } from '../api';
  import {
    featureStack,
    inspectorTarget,
    layer,
    notes
  } from '../stores';

  let filterText = $state('');
  let filterLayer = $state<number | ''>('');
  let loading = $state(false);
  let error = $state<string | null>(null);

  async function reload() {
    loading = true;
    try {
      const r = await listNotes();
      notes.set(r.notes);
      error = null;
    } catch (e) {
      error = (e as Error).message;
    } finally {
      loading = false;
    }
  }

  onMount(reload);

  let filtered = $derived.by(() => {
    const q = filterText.trim().toLowerCase();
    return $notes.filter((n) => {
      if (filterLayer !== '' && n.layer !== filterLayer) return false;
      if (q) {
        const hay = `${n.label} ${n.memo} ${n.feature_idx}`.toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });
  });

  function applyNote(n: { layer: number; feature_idx: number }) {
    layer.set(n.layer);
    featureStack.update((stack) => {
      if (stack.some((f) => f.idx === n.feature_idx)) return stack;
      return [...stack, { idx: n.feature_idx, strength_multiplier: 1.0 }];
    });
    inspectorTarget.set({ layer: n.layer, feature_idx: n.feature_idx });
  }

  async function newFromInspector() {
    const target = $inspectorTarget;
    if (!target) {
      error = 'Open a feature in the Inspector first (right pane), then + New.';
      return;
    }
    const label = window.prompt(
      `Label for feature #${target.feature_idx} on layer ${target.layer}:`,
      `feature ${target.feature_idx}`
    );
    if (!label) return;
    try {
      const created = await createNote({
        layer: target.layer,
        feature_idx: target.feature_idx,
        label,
        memo: ''
      });
      notes.update((ns) => [created, ...ns]);
      error = null;
    } catch (e) {
      error = (e as Error).message;
    }
  }

  async function remove(id: number) {
    if (!window.confirm('Delete this note?')) return;
    try {
      await deleteNote(id);
      notes.update((ns) => ns.filter((n) => n.id !== id));
    } catch (e) {
      error = (e as Error).message;
    }
  }
</script>

<aside class="rounded-lg border border-neutral-800 bg-neutral-900/40 p-4 space-y-3 text-sm">
  <h2 class="text-sm font-semibold text-neutral-300 flex items-center justify-between">
    <span>Notes</span>
    <button
      type="button"
      onclick={newFromInspector}
      class="text-xs px-2 py-0.5 rounded bg-emerald-800 hover:bg-emerald-700 text-white disabled:opacity-50 disabled:cursor-not-allowed"
      disabled={!$inspectorTarget}
      title={$inspectorTarget
        ? `Save feature #${$inspectorTarget.feature_idx} (L${$inspectorTarget.layer}) — currently shown in Inspector`
        : 'Open a feature in the Inspector to enable this'}
    >
      + New
    </button>
  </h2>

  <div class="flex items-center gap-2">
    <input
      type="text"
      placeholder="filter…"
      bind:value={filterText}
      class="flex-1 bg-neutral-900 border border-neutral-700 rounded px-2 py-1 text-xs"
    />
    <input
      type="number"
      placeholder="layer"
      min="0"
      bind:value={filterLayer}
      class="w-16 bg-neutral-900 border border-neutral-700 rounded px-2 py-1 text-xs"
    />
  </div>

  {#if error}
    <div class="text-xs text-red-400 break-words">{error}</div>
  {/if}

  {#if loading}
    <div class="text-xs text-neutral-500">loading…</div>
  {:else if filtered.length === 0}
    <div class="text-xs text-neutral-500 italic">
      No notes yet. Open a feature in the Inspector (right pane), then click "+ New" above for a quick label-only save, or use "+ Save note" inside Inspector for label + memo.
    </div>
  {:else}
    <ul class="space-y-1.5 max-h-[55vh] overflow-y-auto pr-1">
      {#each filtered as n (n.id)}
        <li
          class="rounded border border-neutral-800 bg-neutral-950 p-2 hover:border-emerald-700 cursor-pointer"
        >
          <button
            type="button"
            class="w-full text-left"
            onclick={() => applyNote(n)}
            title="Load into stack & inspector"
          >
            <div class="flex items-baseline justify-between gap-2">
              <span class="text-sm text-emerald-300 font-mono">#{n.feature_idx}</span>
              <span class="text-[10px] text-neutral-500">L{n.layer}</span>
            </div>
            <div class="text-xs text-neutral-200 truncate">{n.label}</div>
            {#if n.memo}
              <div class="text-[11px] text-neutral-500 line-clamp-2">{n.memo}</div>
            {/if}
          </button>
          <div class="flex justify-end mt-1">
            <button
              type="button"
              onclick={() => remove(n.id)}
              class="text-[10px] text-neutral-500 hover:text-red-400"
              title="Delete">×</button
            >
          </div>
        </li>
      {/each}
    </ul>
  {/if}
</aside>
