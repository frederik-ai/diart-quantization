# Created with ChatGPT
# Log 1: https://chatgpt.com/share/67c08fce-b5fc-8001-af76-f36f1f94eb25
# Log 2: https://chatgpt.com/share/67c08e85-2244-8001-823c-d31e7f024e81

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib as mpl

def style_legend_titles_by_setting_position(leg: mpl.legend.Legend, bold: bool = False) -> None:
    """ Style legend "titles"

    A legend entry can be marked as a title by setting visible=False. Titles
    get left-aligned and optionally bolded.
    """
    # matplotlib.offsetbox.HPacker unconditionally adds a pixel of padding
    # around each child.
    hpacker_padding = 2

    for handle, label in zip(leg.legendHandles, leg.texts):
        if not handle.get_visible():
            # See matplotlib.legend.Legend._init_legend_box()
            widths = [leg.handlelength, leg.handletextpad]
            offset_points = sum(leg._fontsize * w for w in widths)
            offset_pixels = leg.figure.canvas.get_renderer().points_to_pixels(offset_points) + hpacker_padding
            label.set_position((-offset_pixels, 0))
            if bold:
                label.set_fontweight('bold')

# Model names
model_names = [
    "pyannote/embedding", 
    "pyannote/wespeaker-voxceleb-resnet34-LM", 
    "speechbrain/spkrec-xvect-voxceleb", 
    "speechbrain/spkrec-ecapa-voxceleb",
]
model_names_quantized = [name + " (Quantized)" for name in model_names]

configurations_pyannote_segmentation = [name + " pyannote/segmentation" for name in model_names+model_names_quantized]
configurations_pyannote_segmentation_3_0 = [name + " pyannote/segmentation-3.0" for name in model_names+model_names_quantized]
configurations = configurations_pyannote_segmentation + configurations_pyannote_segmentation_3_0

# Data: Latency and DER before and after quantization
latency_without_quantization_pyannote_segmentation = [171, 1032, 170, 820]
der_without_quantization_pyannote_segmentation = [36.30, 36.98, 46.20, 36.61]

latency_with_quantization_pyannote_segmentation = [166, 1040, 165, 840]
der_with_quantization_pyannote_segmentation = [36.30, 36.97, 46.20, 36.84]

latency_without_quantization_pyannote_segmentation_3_0 = [156, 1011, 179, 900]
der_without_quantization_pyannote_segmentation_3_0 = [34.57, 31.51, 43.97, 35.97]

latency_with_quantization_pyannote_segmentation_3_0 = [166, 998, 155, 789]
der_with_quantization_pyannote_segmentation_3_0 = [34.65, 31.91, 43.97, 33.43]

plt.figure(figsize=(12, 6))
colors = sns.color_palette("tab10", len(model_names))
markers = {"pyannote/segmentation": "o", "pyannote/segmentation-3.0": "^"}

# Adding labels for the legend
segmentation_marker = plt.scatter([], [], marker="o", color="w", edgecolors="black", s=100, label="pyannote/segmentation")
segmentation_3_marker = plt.scatter([], [], marker="^", color="w", s=100, edgecolors="black", label="pyannote/segmentation-3.0")
quantized_marker = plt.scatter([], [], marker="o", color=colors[0], s=100, alpha=0.5, label="Baseline")
unquantized_marker = plt.scatter([], [], marker="o", color=colors[0], s=100, alpha=1.0, label="Quantized")

# Color of each embedding model
legend_handles = []
for i, model in enumerate(model_names):
    legend_handles.append(plt.scatter([], [], color=colors[i], marker=".", s=100, label=model))

# Plotting the data
for i, model in enumerate(model_names):
    for seg_type, latency_no_q, der_no_q, latency_q, der_q in zip(
        ["pyannote/segmentation", "pyannote/segmentation-3.0"],
        [latency_without_quantization_pyannote_segmentation[i], latency_without_quantization_pyannote_segmentation_3_0[i]],
        [der_without_quantization_pyannote_segmentation[i], der_without_quantization_pyannote_segmentation_3_0[i]],
        [latency_with_quantization_pyannote_segmentation[i], latency_with_quantization_pyannote_segmentation_3_0[i]],
        [der_with_quantization_pyannote_segmentation[i], der_with_quantization_pyannote_segmentation_3_0[i]]):
        
        if np.isnan(der_no_q) or np.isnan(der_q):
            continue
        
        plt.scatter(latency_no_q, der_no_q, color=colors[i], marker=markers[seg_type], s=150, edgecolor='black', alpha=0.5, label=f"{model} ({seg_type})")
        plt.scatter(latency_q, der_q, color=colors[i], marker=markers[seg_type], s=150, edgecolor='black', alpha=1.0)

        if abs(latency_no_q - latency_q) > 20 or abs(der_no_q - der_q) > 2:
            arrow = FancyArrowPatch((latency_no_q, der_no_q), (latency_q, der_q),
                                    connectionstyle="arc3,rad=0.3", arrowstyle="->",
                                    color=colors[i], mutation_scale=15, lw=2)
            plt.gca().add_patch(arrow)

# Create custom legend handles for each section
handles_quantization = [
    mpatches.Patch(visible=False, label='Quantization'),
    plt.scatter([], [], color=colors[0], marker="o", s=100, alpha=0.5, label="Baseline"),
    plt.scatter([], [], color=colors[0], marker="o", s=100, alpha=1.0, label="Quantized")
]

handles_segmentation = [
    mpatches.Patch(visible=False, label='Segmentation Models'),
    plt.scatter([], [], color="w", marker="o", edgecolors="black", s=100, label='pyannote/segmentation'),
    plt.scatter([], [], color="w", marker="^", edgecolors="black", s=100, label='pyannote/segmentation-3.0')
]

handles_models = [
    mpatches.Patch(visible=False, label='Embedding Models'),
] + [plt.scatter([], [], color=colors[i], marker=".", s=100, label=model) for i, model in enumerate(model_names)]

# Combining all handles into one list with blank entries to simulate sections
all_handles = handles_quantization + [mpatches.Patch(visible=False, label="")] + handles_segmentation + [mpatches.Patch(visible=False, label="")] + handles_models

plt.xlabel("Latency (ms)", fontsize=14)
plt.ylabel("Diarization Error Rate (%)", fontsize=14)
plt.xscale("log")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

leg = plt.legend(handles=all_handles, labels=[handle.get_label() for handle in all_handles], fontsize=12, loc="upper center")
style_legend_titles_by_setting_position(leg, bold=False)

# Adjust layout
plt.tight_layout()
plt.savefig("teaser_figure.png", dpi=600)  # Higher DPI for publication
plt.show()