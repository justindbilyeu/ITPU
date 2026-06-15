"""Send the ITPU hardware schematic to Claude for architectural analysis.

Usage:
    python ask_claude.py [image_path]

Defaults to the schematic PNG in the repo root.
Streams the analysis to stdout and writes schematic_analysis.txt.
Requires ANTHROPIC_API_KEY in the environment.
"""
import base64
import sys
from pathlib import Path

import anthropic

DEFAULT_IMAGE = Path(__file__).parent / "4F4F1C25-07C8-4C70-83E0-7CF0A892819F.PNG"

SYSTEM_PROMPT = (
    "You are an expert FPGA architect and information-theoretic systems designer "
    "reviewing a hardware block diagram for the ITPU (Information-Theoretic Processing Unit).\n\n"
    "Context:\n"
    "- ITPU is a dedicated silicon accelerator for mutual information, entropy, and k-NN "
    "statistics — workloads that are branch-heavy, memory-bound, and poorly suited to "
    "matrix-multiply hardware.\n"
    "- The software SDK ships first (Python, tested and calibrated). The FPGA pathfinder "
    "(R2) is next; the silicon target is R3.\n"
    "- Three core kernels, with measured software performance baselines:\n"
    "    1. KSG (Kraskov–Stögbauer–Grassberger) k-NN MI: 230 ns/sample target\n"
    "    2. Histogram MI: 150 ns/sample target\n"
    "    3. IAAFT surrogate generation: 1.2 µs/surrogate target\n"
    "- The software stack uses: Chebyshev (L∞) metric for KSG k-NN search, rfft/irfft "
    "for IAAFT, and 2D histogram binning for MI. These must map cleanly to hardware primitives.\n"
    "- The permutation p-value formula is locked: p = (#{null ≥ observed} + 1) / (n + 1). "
    "Hardware must preserve this exact arithmetic.\n"
    "- Target domains: BCI/EEG real-time MI, medical imaging registration, causal ML.\n\n"
    "Your role: review the schematic with the eye of someone who will eventually tape this out. "
    "Be direct about design quality, bottlenecks, and risks."
)

USER_PROMPT = (
    "Please analyze this ITPU FPGA architecture schematic. I want your honest assessment on:\n\n"
    "1. **Pipeline throughput** — Do the proposed datapath widths and clock domains support "
    "the KSG (230 ns), Histogram (150 ns), and IAAFT (1.2 µs) targets? Where are the "
    "throughput ceilings?\n\n"
    "2. **Memory bandwidth** — What are the on-chip SRAM and off-chip DRAM bandwidth "
    "requirements? Are there obvious bandwidth bottlenecks for the k-NN search or FFT stages?\n\n"
    "3. **Clock domain and timing** — Are there concerning CDC (clock domain crossing) "
    "boundaries? Is the pipeline depth reasonable for timing closure at the target frequency?\n\n"
    "4. **Kernel-to-hardware mapping** — How well does the schematic map to the three "
    "software kernels (Chebyshev k-NN, histogram accumulation, rfft/irfft)? Are there "
    "structural mismatches that will require algorithmic changes?\n\n"
    "5. **R2 pathfinder readiness** — What would you change before committing this to an "
    "FPGA development board? What are the one or two highest-risk items that could sink R2?\n\n"
    "Be specific. Point to structures in the diagram when you can."
)


def main(image_path: Path) -> None:
    if not image_path.exists():
        print(f"Error: image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    client = anthropic.Anthropic()

    print(f"Image: {image_path.name}")
    print(f"Model: claude-sonnet-4-6  |  thinking: adaptive")
    print("=" * 72)

    text_chunks: list[str] = []
    in_thinking = False

    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_data,
                        },
                    },
                    {"type": "text", "text": USER_PROMPT},
                ],
            }
        ],
    ) as stream:
        for event in stream:
            if event.type == "content_block_start":
                if event.content_block.type == "thinking":
                    print("\n[Thinking]\n", flush=True)
                    in_thinking = True
                elif event.content_block.type == "text":
                    if in_thinking:
                        print("\n[Analysis]\n", flush=True)
                    in_thinking = False
            elif event.type == "content_block_delta":
                if event.delta.type == "thinking_delta":
                    print(event.delta.thinking, end="", flush=True)
                elif event.delta.type == "text_delta":
                    print(event.delta.text, end="", flush=True)
                    text_chunks.append(event.delta.text)
        final = stream.get_final_message()

    print("\n" + "=" * 72)

    u = final.usage
    print(f"\nUsage:")
    print(f"  input_tokens:                {u.input_tokens:,}")
    print(f"  cache_creation_input_tokens: {u.cache_creation_input_tokens:,}")
    print(f"  cache_read_input_tokens:     {u.cache_read_input_tokens:,}")
    print(f"  output_tokens:               {u.output_tokens:,}")

    out_path = Path(__file__).parent / "schematic_analysis.txt"
    out_path.write_text("".join(text_chunks), encoding="utf-8")
    print(f"\nAnalysis written to {out_path.name}")


if __name__ == "__main__":
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE
    main(image_path)
