import io
from typing import Dict, List

import streamlit as st
from PIL import Image

from pipelines import run_single, run_custom
from registry import TECHNIQUES


PRESET_PIPELINES: Dict[str, Dict] = {
    "Dehaze → Retinex → CLAHE": {
        "techniques": ["dehaze", "retinex", "clahe"],
        "params": {
            "dehaze": {"omega": 0.95, "t0": 0.1, "radius": 15},
            "retinex": {"sigmas": [15, 80, 250], "alpha": 125},
            "clahe": {"clip_limit": 2.0, "tile_size": 8},
        },
    },
    "Denoise → White balance → Gamma": {
        "techniques": ["denoise", "white_balance", "gamma_correction"],
        "params": {
            "denoise": {"h": 10, "h_for_color": 10, "template_window": 7, "search_window": 21},
            "white_balance": {"percent": 1.0},
            "gamma_correction": {"gamma": 1.2},
        },
    },
    "Super-resolution → CLAHE": {
        "techniques": ["superres", "clahe"],
        "params": {
            "superres": {"scale": 2.0},
            "clahe": {"clip_limit": 2.0, "tile_size": 8},
        },
    },
}


def _bytes_to_pil(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def main() -> None:
    st.set_page_config(page_title="Image Enhancement", layout="wide")
    st.title("Image Enhancement")
    st.write(
        "Upload an image and run either a **single technique** or a **preset pipeline** "
        "without writing any code."
    )

    mode = st.sidebar.selectbox(
        "Mode",
        options=["Single technique", "Preset pipeline"],
    )

    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])

    col1, col2 = st.columns(2)

    if uploaded_file is not None:
        # Cache bytes in session state; file buffer can return empty bytes after reruns
        cache_key = f"{uploaded_file.name}_{uploaded_file.size}"
        if (
            "uploaded_bytes" not in st.session_state
            or st.session_state.get("uploaded_cache_key") != cache_key
        ):
            uploaded_file.seek(0)
            st.session_state.uploaded_bytes = uploaded_file.read()
            st.session_state.uploaded_cache_key = cache_key
        orig_bytes = st.session_state.uploaded_bytes
        with col1:
            st.subheader("Original")
            st.image(_bytes_to_pil(orig_bytes), use_column_width=True)
    else:
        st.info("Upload an image to get started.")
        return

    if mode == "Single technique":
        technique_names: List[str] = sorted(TECHNIQUES.keys())
        technique = st.sidebar.selectbox("Technique", options=technique_names)

        params: Dict[str, Dict] = {}
        with st.sidebar.expander("Technique parameters", expanded=True):
            if technique == "superres":
                scale = st.slider("Scale", 1.0, 4.0, 2.0, 0.1)
                params[technique] = {"scale": scale}
            elif technique == "retinex":
                alpha = st.slider("Alpha (color restoration)", 50.0, 200.0, 125.0, 5.0)
                params[technique] = {"alpha": alpha}
            elif technique == "dehaze":
                omega = st.slider("Omega (haze removal strength)", 0.5, 1.0, 0.95, 0.01)
                t0 = st.slider("Minimum transmission t0", 0.05, 0.5, 0.1, 0.01)
                radius = st.slider("Patch radius", 3, 31, 15, 2)
                params[technique] = {"omega": omega, "t0": t0, "radius": radius}
            elif technique == "denoise":
                h = st.slider("Luminance strength (h)", 1.0, 30.0, 10.0, 1.0)
                h_color = st.slider("Color strength", 1.0, 30.0, 10.0, 1.0)
                t_win = st.slider("Template window", 3, 11, 7, 2)
                s_win = st.slider("Search window", 7, 31, 21, 2)
                params[technique] = {
                    "h": h,
                    "h_for_color": h_color,
                    "template_window": t_win,
                    "search_window": s_win,
                }
            elif technique == "white_balance":
                percent = st.slider("Clipping percent", 0.0, 20.0, 1.0, 0.5)
                params[technique] = {"percent": percent}
            elif technique == "gamma_correction":
                gamma = st.slider("Gamma", 0.5, 3.0, 1.2, 0.1)
                params[technique] = {"gamma": gamma}
            elif technique == "clahe":
                clip = st.slider("Clip limit", 1.0, 5.0, 2.0, 0.1)
                tile = st.slider("Tile size", 2, 32, 8, 2)
                params[technique] = {"clip_limit": clip, "tile_size": tile}
            else:
                params[technique] = {}

        run_button = st.button("Run single technique")
        if run_button:
            with st.spinner("Running technique..."):
                out_bytes = run_single(orig_bytes, technique, params)
            with col2:
                st.subheader("Result")
                st.image(_bytes_to_pil(out_bytes), use_column_width=True)
                st.download_button(
                    "Download result",
                    data=out_bytes,
                    file_name=f"enhanced_{uploaded_file.name}",
                    mime="image/png",
                )

    else:  # Preset pipeline
        pipeline_name = st.sidebar.selectbox(
            "Pipeline",
            options=list(PRESET_PIPELINES.keys()),
        )
        pipeline = PRESET_PIPELINES[pipeline_name]
        techniques = pipeline["techniques"]
        params = pipeline["params"]

        st.sidebar.markdown("**Pipeline steps:**")
        for idx, t in enumerate(techniques, start=1):
            st.sidebar.write(f"{idx}. `{t}`")

        run_button = st.button(f"Run pipeline: {pipeline_name}")
        if run_button:
            with st.spinner("Running pipeline..."):
                out_bytes = run_custom(orig_bytes, techniques, params)
            with col2:
                st.subheader("Result")
                st.image(_bytes_to_pil(out_bytes), use_column_width=True)
                st.download_button(
                    "Download result",
                    data=out_bytes,
                    file_name=f"pipeline_{uploaded_file.name}",
                    mime="image/png",
                )


if __name__ == "__main__":
    main()

