"""Generate all figures for the Two Refusal Directions paper."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

OUTDIR = "figures"


# ─── Figure 1: Compliance vs Training Examples (data ablation curve) ─────────

def fig1_data_ablation():
    examples = [1, 3, 5, 10, 15, 29]
    compliance = [26.7, 66.7, 66.7, 86.7, 86.7, 100.0]
    thinking = [80.0, 86.7, 100.0, 100.0, 100.0, 100.0]

    fig, ax1 = plt.subplots(figsize=(6, 4))

    color1 = '#2563eb'
    color2 = '#9333ea'

    ax1.plot(examples, compliance, 'o-', color=color1, linewidth=2.5,
             markersize=8, label='Compliance rate', zorder=5)
    ax1.fill_between(examples, compliance, alpha=0.08, color=color1)
    ax1.set_xlabel('Number of training examples')
    ax1.set_ylabel('Compliance rate (%)', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0, 105)
    ax1.set_xlim(0, 31)

    # Phase annotations
    ax1.annotate('Phase 1:\nformat\nlearning',
                xy=(2, 46), fontsize=9, color='#666',
                ha='center', style='italic')
    ax1.annotate('Phase 2:\ncategory\ncoverage',
                xy=(7, 46), fontsize=9, color='#666',
                ha='center', style='italic')
    ax1.annotate('', xy=(3.5, 40), xytext=(0.5, 40),
                arrowprops=dict(arrowstyle='<->', color='#999', lw=1))
    ax1.annotate('', xy=(10.5, 40), xytext=(3.5, 40),
                arrowprops=dict(arrowstyle='<->', color='#999', lw=1))

    # Knee annotation
    ax1.axhline(y=80, color='#ccc', linestyle='--', linewidth=0.8)
    ax1.text(30, 81.5, '80% threshold', ha='right', fontsize=8, color='#999')

    ax2 = ax1.twinx()
    ax2.plot(examples, thinking, 's--', color=color2, linewidth=1.5,
             markersize=6, alpha=0.7, label='Thinking rate')
    ax2.set_ylabel('Thinking rate (%)', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, 105)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right',
               framealpha=0.9)

    plt.title('Compliance vs. Training Examples (Qwen3.5-27B)')
    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/fig1_data_ablation.pdf')
    fig.savefig(f'{OUTDIR}/fig1_data_ablation.png')
    plt.close()
    print('  fig1_data_ablation')


# ─── Figure 2: Harmfulness probe accuracy by layer ──────────────────────────

def fig2_harmfulness_probes():
    layers = [0, 1, 4, 8, 12, 16, 20, 24, 27, 28, 32, 33, 36, 40, 44, 48, 52, 56, 60, 63]
    # Both base and compliance have identical accuracies
    base_acc = [75.0, 81.7, 91.7, 98.3, 100, 100, 100, 100, 100, 100,
                100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    comp_acc = base_acc[:]  # identical

    fig, ax = plt.subplots(figsize=(7, 3.5))

    ax.plot(layers, base_acc, 'o-', color='#2563eb', linewidth=2, markersize=5,
            label='Base model', alpha=0.9)
    ax.plot(layers, comp_acc, 's--', color='#dc2626', linewidth=2, markersize=5,
            label='Compliance model', alpha=0.7)

    ax.fill_between(layers, base_acc, alpha=0.06, color='#2563eb')
    ax.set_xlabel('Layer index')
    ax.set_ylabel('Probe accuracy (%)')
    ax.set_ylim(50, 102)
    ax.set_xlim(-1, 65)

    # Annotate the key finding
    ax.annotate('$\\Delta = 0.000$ at every layer',
                xy=(35, 92), fontsize=11, fontweight='bold',
                color='#dc2626', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#fef2f2',
                         edgecolor='#dc2626', alpha=0.9))

    ax.axhline(y=100, color='#ddd', linestyle='-', linewidth=0.5)
    ax.legend(loc='lower right')
    plt.title('Harmfulness Probe: Representation Survives Compliance Tuning')
    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/fig2_harmfulness_probes.pdf')
    fig.savefig(f'{OUTDIR}/fig2_harmfulness_probes.png')
    plt.close()
    print('  fig2_harmfulness_probes')


# ─── Figure 3: DeltaNet gate probe accuracy by layer ────────────────────────

def fig3_deltanet_gates():
    layers = [0, 1, 8, 16, 24, 33, 48, 60]
    accuracy = [43.3, 53.3, 80.0, 93.3, 93.3, 100.0, 86.7, 100.0]
    load_bearing = [True, False, False, False, False, True, False, False]

    fig, ax = plt.subplots(figsize=(6, 4))

    colors = ['#dc2626' if lb else '#2563eb' for lb in load_bearing]
    bars = ax.bar(range(len(layers)), accuracy, color=colors, alpha=0.85,
                  edgecolor='white', linewidth=0.5)

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([str(l) for l in layers])
    ax.set_xlabel('Layer index')
    ax.set_ylabel('Gate probe accuracy (%)')
    ax.set_ylim(0, 110)

    ax.axhline(y=50, color='#ccc', linestyle='--', linewidth=0.8)
    ax.text(7.5, 51.5, 'chance', ha='right', fontsize=8, color='#999')

    # Annotations for the key asymmetry
    ax.annotate('weight-level\n(random)',
                xy=(0, 43.3), xytext=(0, 15),
                fontsize=8, color='#dc2626',
                arrowprops=dict(arrowstyle='->', color='#dc2626', lw=1),
                ha='center')
    ax.annotate('dynamics-level\n(perfect)',
                xy=(5, 100), xytext=(6.2, 80),
                fontsize=8, color='#dc2626',
                arrowprops=dict(arrowstyle='->', color='#dc2626', lw=1),
                ha='center')

    lb_patch = mpatches.Patch(color='#dc2626', alpha=0.85, label='Load-bearing layer')
    other_patch = mpatches.Patch(color='#2563eb', alpha=0.85, label='Other DeltaNet layer')
    ax.legend(handles=[lb_patch, other_patch], loc='center left')

    plt.title('DeltaNet Attention Dynamics Encode Harmfulness')
    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/fig3_deltanet_gates.pdf')
    fig.savefig(f'{OUTDIR}/fig3_deltanet_gates.png')
    plt.close()
    print('  fig3_deltanet_gates')


# ─── Figure 4: Cross-model comparison ───────────────────────────────────────

def fig4_cross_model():
    models = ['Qwen3.5-27B\n(hybrid)', 'QwQ-32B\n(standard)', 'DeepSeek-R1\n-Distill-32B']
    compliance = [100, 86.7, 100]
    loss = [0.13, 1.17, 1.63]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

    colors = ['#2563eb', '#9333ea', '#059669']

    # Compliance bars
    bars1 = ax1.bar(range(len(models)), compliance, color=colors, alpha=0.85,
                    edgecolor='white', linewidth=0.5)
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, fontsize=9)
    ax1.set_ylabel('Compliance rate (%)')
    ax1.set_ylim(0, 115)
    ax1.axhline(y=80, color='#ccc', linestyle='--', linewidth=0.8)
    ax1.text(2.4, 81.5, '80%', fontsize=8, color='#999')
    for i, v in enumerate(compliance):
        ax1.text(i, v + 2, f'{v}%', ha='center', fontsize=10, fontweight='bold')
    ax1.set_title('(a) Compliance Rate')

    # Training loss bars
    bars2 = ax2.bar(range(len(models)), loss, color=colors, alpha=0.85,
                    edgecolor='white', linewidth=0.5)
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, fontsize=9)
    ax2.set_ylabel('Final training loss')
    ax2.set_ylim(0, 2.0)
    for i, v in enumerate(loss):
        ax2.text(i, v + 0.05, f'{v:.2f}', ha='center', fontsize=10)
    ax2.set_title('(b) Training Loss')

    fig.suptitle('Cross-Model Transfer (same 29 training examples)', fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/fig4_cross_model.pdf')
    fig.savefig(f'{OUTDIR}/fig4_cross_model.png')
    plt.close()
    print('  fig4_cross_model')


# ─── Figure 5: Method comparison (all approaches on Qwen3.5-27B) ────────────

def fig5_method_comparison():
    methods = [
        'Baseline',
        'Std. abliteration',
        'Abliteration\n+ think strip',
        'Nuclear',
        '$\\mathbf{d}_{static}$\nremoval',
        '$\\mathbf{d}_{thinking}$\nremoval',
        'QLoRA SFT\n(this work)',
    ]
    compliance = [0, 30, 35, 0, 100, 100, 100]
    colors = ['#94a3b8', '#f59e0b', '#f59e0b', '#ef4444',
              '#3b82f6', '#3b82f6', '#22c55e']

    fig, ax = plt.subplots(figsize=(8, 4))

    bars = ax.barh(range(len(methods)), compliance, color=colors, alpha=0.85,
                   edgecolor='white', linewidth=0.5, height=0.7)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xlabel('Compliance rate (%)')
    ax.set_xlim(0, 115)
    ax.invert_yaxis()

    for i, v in enumerate(compliance):
        if v > 0:
            ax.text(v + 2, i, f'{v}%', va='center', fontsize=10, fontweight='bold')
        else:
            ax.text(2, i, 'corrupted' if i == 3 else '0%', va='center',
                    fontsize=9, color='#999')

    ax.axvline(x=80, color='#ccc', linestyle='--', linewidth=0.8)

    plt.title('Safety Removal Methods on Qwen3.5-27B')
    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/fig5_method_comparison.pdf')
    fig.savefig(f'{OUTDIR}/fig5_method_comparison.png')
    plt.close()
    print('  fig5_method_comparison')


# ─── Figure 6: Multi-turn robustness heatmap ────────────────────────────────

def fig6_multiturn():
    scenarios = ['Escalation', 'Topic pivot', 'Authority', 'Roleplay', 'Sustained', 'Meta manip.']
    turns = ['Turn 1', 'Turn 2', 'Turn 3', 'Turn 4', 'Turn 5']

    # 1 = compliant, 0 = refused
    data = np.array([
        [1, 1, 1, 1, 1],  # escalation
        [1, 1, 1, 1, 1],  # topic pivot
        [1, 1, 1, 1, 1],  # authority
        [1, 1, 0, 0, 1],  # roleplay
        [1, 1, 1, 1, 1],  # sustained
        [1, 0, 0, 1, 1],  # meta manipulation
    ])

    fig, ax = plt.subplots(figsize=(6, 3.5))

    cmap = matplotlib.colors.ListedColormap(['#fecaca', '#bbf7d0'])
    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(turns)))
    ax.set_xticklabels(turns, fontsize=9)
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels(scenarios, fontsize=9)

    for i in range(len(scenarios)):
        for j in range(len(turns)):
            text = 'C' if data[i, j] == 1 else 'R'
            color = '#166534' if data[i, j] == 1 else '#991b1b'
            ax.text(j, i, text, ha='center', va='center',
                    fontsize=10, fontweight='bold', color=color)

    # Grid
    ax.set_xticks(np.arange(-.5, len(turns), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(scenarios), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=2)
    ax.tick_params(which='minor', size=0)

    comply = mpatches.Patch(color='#bbf7d0', label='Comply (C)')
    refuse = mpatches.Patch(color='#fecaca', label='Refuse (R)')
    ax.legend(handles=[comply, refuse], loc='upper right',
              bbox_to_anchor=(1.25, 1), fontsize=9)

    plt.title('Multi-Turn Adversarial Robustness', pad=10)
    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/fig6_multiturn.pdf')
    fig.savefig(f'{OUTDIR}/fig6_multiturn.png')
    plt.close()
    print('  fig6_multiturn')


# ─── Figure 7: Training loss curve ──────────────────────────────────────────

def fig7_loss_curve():
    steps = list(range(1, 25))
    epochs = [s * 3.0 / 24 for s in steps]
    loss = [1.498, 1.107, 1.145, 0.980, 1.016, 0.945, 0.915, 0.906,
            0.611, 0.469, 0.524, 0.521, 0.641, 0.506, 0.381, 0.385,
            0.271, 0.200, 0.234, 0.184, 0.153, 0.140, 0.131, 0.131]

    fig, ax = plt.subplots(figsize=(6, 3))

    ax.plot(epochs, loss, '-', color='#2563eb', linewidth=2, alpha=0.9)
    ax.fill_between(epochs, loss, alpha=0.06, color='#2563eb')

    # Epoch boundaries
    for e in [1.0, 2.0]:
        ax.axvline(x=e, color='#e5e7eb', linestyle='-', linewidth=0.8)
        ax.text(e + 0.02, 1.45, f'Epoch {int(e)}', fontsize=8, color='#999')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training loss')
    ax.set_ylim(0, 1.6)
    ax.set_xlim(0, 3.1)

    plt.title('QLoRA Training Loss (29 examples, Qwen3.5-27B)')
    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/fig7_loss_curve.pdf')
    fig.savefig(f'{OUTDIR}/fig7_loss_curve.png')
    plt.close()
    print('  fig7_loss_curve')


# ─── Figure 8: Abliteration cliff (phase transition) ────────────────────────

def fig8_abliteration_cliff():
    # Regime 1: preserved but 0% compliance
    regime1_ppl = [8.03, 6.96, 7.64, 7.45, 6.76, 7.38, 7.31]
    regime1_compliance = [0, 0, 0, 0, 0, 0, 0]
    regime1_labels = [
        'D_think, down', 'D_think, down r=0.5', 'D_think, MLP',
        'D_think, MLP+o', 'D_stat, down', 'D_stat, MLP', 'D_stat, MLP+o'
    ]

    # Regime 2: corrupted (ppl → ∞, compliance undefined)
    regime2_ppl = [50, 50, 50]  # represent as high
    regime2_compliance = [0, 0, 0]
    regime2_labels = ['D_stat, all', 'D_think, all', 'Combined, all']

    # Successful methods for contrast
    success_ppl = [2.65, 2.71]  # QLoRA SFT, head suppression (approx)
    success_compliance = [100, 75]
    success_labels = ['QLoRA SFT', 'Head supp.']

    fig, ax = plt.subplots(figsize=(7, 5))

    # Regime 1
    ax.scatter(regime1_compliance, regime1_ppl, c='#f59e0b', s=80,
               marker='o', zorder=5, label='Regime 1: preserved, 0% compliance',
               edgecolors='#b45309', linewidths=0.8)

    # Regime 2
    ax.scatter(regime2_compliance, regime2_ppl, c='#ef4444', s=120,
               marker='X', zorder=5, label='Regime 2: corrupted (ppl → ∞)',
               edgecolors='#991b1b', linewidths=0.8)

    # Successful methods
    ax.scatter(success_compliance, success_ppl, c='#22c55e', s=120,
               marker='*', zorder=5, label='Successful methods',
               edgecolors='#166534', linewidths=0.8)
    for i, label in enumerate(success_labels):
        ax.annotate(label, (success_compliance[i], success_ppl[i]),
                    textcoords='offset points', xytext=(10, 5),
                    fontsize=8, color='#166534')

    # Baseline
    ax.axhline(y=2.62, color='#94a3b8', linestyle=':', linewidth=1)
    ax.text(50, 2.2, 'Baseline ppl = 2.62', fontsize=8, color='#94a3b8', ha='center')

    # Dead zone annotation
    ax.fill_between([5, 95], [0, 0], [55, 55], alpha=0.03, color='#94a3b8')
    ax.annotate('No viable\nintermediate\nregime',
                xy=(50, 25), fontsize=11, color='#94a3b8',
                ha='center', va='center', style='italic')

    # Arrows showing the two regimes
    ax.annotate('', xy=(0, 10), xytext=(0, 42),
                arrowprops=dict(arrowstyle='<->', color='#f59e0b', lw=1.5))
    ax.text(-3, 26, 'Regime 1', fontsize=8, color='#b45309',
            ha='right', rotation=90, va='center')

    ax.set_xlabel('Compliance rate (%)')
    ax.set_ylabel('Perplexity')
    ax.set_xlim(-10, 110)
    ax.set_ylim(0, 55)
    ax.legend(loc='upper right', fontsize=9)

    plt.title('The Abliteration Cliff: Phase Transition in Weight Projection')
    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/fig8_abliteration_cliff.pdf')
    fig.savefig(f'{OUTDIR}/fig8_abliteration_cliff.png')
    plt.close()
    print('  fig8_abliteration_cliff')


if __name__ == '__main__':
    print('Generating figures...')
    fig1_data_ablation()
    fig2_harmfulness_probes()
    fig3_deltanet_gates()
    fig4_cross_model()
    fig5_method_comparison()
    fig6_multiturn()
    fig7_loss_curve()
    fig8_abliteration_cliff()
    print('Done! 8 figures saved to figures/')
