"""
Mislabel Image Reviewer - Streamlit App
========================================
Review images flagged as potential mislabels by multi-model consensus
and decide whether to relabel, remove, or keep each image.

Usage:
    cd food_classifier_project/cleaning_results
    streamlit run mislabel_reviewer.py

Features:
- Two tabs: Tier 1 (3/3 consensus) and Tier 2 (2/3 consensus)
- Images grouped by confusion pair (label A <-> models say B)
- Per-image actions: Keep, Remove, or Relabel to predicted class
- Shows model confidence and agreement count
- Filter by confusion pair, class, or review status
- Export decisions to CSV for batch processing
"""

import streamlit as st
import pandas as pd
from PIL import Image
from pathlib import Path
import json
import shutil
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Mislabel Reviewer",
    page_icon="🏷️",
    layout="wide",
)

# ============================================================
# Configuration
# ============================================================

DEFAULT_TIER1_CSV = "05a_mislabels_3of3_consensus.csv"
DEFAULT_TIER2_CSV = "05b_mislabels_2of3_consensus.csv"
PROGRESS_PATH = "mislabel_review_progress.json"
EXPORT_PATH = "mislabel_decisions.csv"

# Actions a reviewer can take on each image
ACTION_KEEP = "keep"        # label is correct, models are wrong
ACTION_REMOVE = "remove"    # image is junk / ambiguous, remove it
ACTION_RELABEL = "relabel"  # move to the class the models predicted

ACTION_LABELS = {
    ACTION_KEEP: "✅ Keep (label correct)",
    ACTION_REMOVE: "🗑️ Remove",
    ACTION_RELABEL: "🔄 Relabel to predicted",
}


# ============================================================
# Helper Functions
# ============================================================

@st.cache_data
def load_mislabel_csv(path: str):
    """Load a mislabel CSV and build groups by confusion pair hash."""
    if not Path(path).exists():
        return None, []

    df = pd.read_csv(path)
    groups = []
    for hash_val, group_df in df.groupby("md5"):
        records = group_df.to_dict("records")
        gt_labels = list(group_df["label"].unique())
        pred_labels = list(group_df["predicted_label"].unique())
        groups.append({
            "hash": hash_val,
            "images": records,
            "gt_labels": gt_labels,
            "pred_labels": pred_labels,
            "is_cross_class": group_df["label"].nunique() > 1 or group_df["predicted_label"].nunique() > 1,
            "size": len(records),
        })
    # Sort: largest groups first
    groups.sort(key=lambda g: -g["size"])
    return df, groups


def load_image_safe(path: str, max_size: int = 400):
    """Load image with error handling and resize for display."""
    try:
        clean_path = path.replace("\\", "/")
        if clean_path.startswith("../"):
            # CSV paths are relative to food_classifier_project/
            # App runs from food_classifier_project/cleaning_results/
            # Resolve __file__ first so .parent.parent navigates correctly
            base_dir = Path(__file__).resolve().parent.parent
            clean_path = (base_dir / clean_path).resolve()
        img = Image.open(clean_path).convert("RGB")
        ratio = max_size / max(img.size)
        if ratio < 1:
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        return img
    except Exception:
        return None


def load_progress():
    """Load review progress from disk."""
    if Path(PROGRESS_PATH).exists():
        with open(PROGRESS_PATH, "r") as f:
            return json.load(f)
    return {"decisions": {}, "reviewed_hashes": []}


def save_progress(progress):
    """Save review progress to disk."""
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f, indent=2)


def get_decision_key(path: str) -> str:
    """Unique key for an image decision, based on its path."""
    return path.replace("\\", "/")


def export_decisions(progress):
    """Export all decisions to a CSV."""
    rows = []
    for path, info in progress["decisions"].items():
        rows.append({
            "path": path,
            "original_label": info["original_label"],
            "predicted_label": info["predicted_label"],
            "action": info["action"],
            "num_models_agree": info.get("num_models_agree", ""),
            "convnext_confidence": info.get("convnext_confidence", ""),
            "tier": info.get("tier", ""),
        })
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(EXPORT_PATH, index=False)
        return df
    return None


# ============================================================
# Display helpers
# ============================================================

def render_group(group, tier_label, progress, images_per_row=4):
    """Render a single confusion-pair group with per-image actions."""
    images = group["images"]
    hash_val = group["hash"]

    # Group header
    is_reviewed = hash_val in progress["reviewed_hashes"]
    status_badge = "✅" if is_reviewed else "⏳"

    gt_str = ", ".join(group["gt_labels"])
    pred_str = ", ".join(group["pred_labels"])
    st.markdown(
        f"**Ground truth:** {gt_str} → **Models predict:** {pred_str} "
        f"| {len(images)} images | {status_badge}"
    )

    # Render images in rows
    selections = {}
    for row_start in range(0, len(images), images_per_row):
        row_images = images[row_start : row_start + images_per_row]
        cols = st.columns(images_per_row)

        for col_i, img_info in enumerate(row_images):
            idx = row_start + col_i
            dkey = get_decision_key(img_info["path"])
            existing = progress["decisions"].get(dkey, {})
            existing_action = existing.get("action", None)

            with cols[col_i]:
                # Image
                img = load_image_safe(img_info["path"])
                if img:
                    st.image(img, use_container_width=True)
                else:
                    st.error("Cannot load image")
                    with st.expander("Path"):
                        st.code(img_info["path"])

                # Info line
                conf = img_info.get("convnext_confidence", 0)
                n_agree = img_info.get("num_models_agree", "?")
                st.caption(
                    f"Label: **{img_info['label']}** → Pred: **{img_info['predicted_label']}**"
                )
                st.caption(f"Models: {n_agree}/3 | Conf: {conf:.2%}")
                st.caption(Path(img_info["path"]).name[:35])

                # Action radio
                action_options = list(ACTION_LABELS.values())
                # Determine default index
                if existing_action == ACTION_KEEP:
                    default_idx = 0
                elif existing_action == ACTION_REMOVE:
                    default_idx = 1
                elif existing_action == ACTION_RELABEL:
                    default_idx = 2
                else:
                    default_idx = 2  # default to relabel (models likely right)

                chosen = st.radio(
                    "Action",
                    action_options,
                    index=default_idx,
                    key=f"{tier_label}_{hash_val}_{idx}",
                    label_visibility="collapsed",
                )

                # Map back to action key
                reverse_map = {v: k for k, v in ACTION_LABELS.items()}
                selections[idx] = {
                    "path": img_info["path"],
                    "action": reverse_map[chosen],
                    "original_label": img_info["label"],
                    "predicted_label": img_info["predicted_label"],
                    "num_models_agree": img_info.get("num_models_agree", ""),
                    "convnext_confidence": img_info.get("convnext_confidence", ""),
                    "tier": tier_label,
                }

    return selections


def render_tab(tier_label, groups, progress, nav_key, filter_key):
    """Render a full tier tab with filtering, navigation, and actions."""
    if not groups:
        st.info(f"No {tier_label} groups found. Check the CSV path in the sidebar.")
        return

    # Filter bar
    col_f1, col_f2, col_f3 = st.columns([2, 2, 1])
    with col_f1:
        filter_opt = st.selectbox(
            "Filter",
            ["All", "Not reviewed", "Reviewed"],
            key=f"{filter_key}_filter",
        )
    with col_f2:
        # Optional class filter
        all_classes = sorted(set(
            c for g in groups for c in g["gt_labels"] + g["pred_labels"]
        ))
        class_filter = st.selectbox(
            "Filter by class",
            ["All classes"] + all_classes,
            key=f"{filter_key}_class",
        )

    # Apply filters
    filtered = groups
    if filter_opt == "Not reviewed":
        filtered = [g for g in filtered if g["hash"] not in progress["reviewed_hashes"]]
    elif filter_opt == "Reviewed":
        filtered = [g for g in filtered if g["hash"] in progress["reviewed_hashes"]]

    if class_filter != "All classes":
        filtered = [
            g for g in filtered
            if class_filter in g["gt_labels"] or class_filter in g["pred_labels"]
        ]

    with col_f2:
        st.caption(f"{len(filtered)} groups")

    if not filtered:
        st.info("No groups match the current filters.")
        return

    # Navigation
    if nav_key not in st.session_state:
        st.session_state[nav_key] = 0
    st.session_state[nav_key] = max(0, min(st.session_state[nav_key], len(filtered) - 1))

    with col_f3:
        group_idx = st.number_input(
            "Group #",
            min_value=1,
            max_value=len(filtered),
            value=st.session_state[nav_key] + 1,
            key=f"{filter_key}_idx",
        ) - 1
        st.session_state[nav_key] = group_idx

    group = filtered[group_idx]

    st.markdown(f"### Group {group_idx + 1} / {len(filtered)}")

    # Render the group
    selections = render_group(group, tier_label, progress)

    # Action buttons
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("⬅️ Previous", disabled=group_idx == 0, key=f"{filter_key}_prev", use_container_width=True):
            st.session_state[nav_key] = group_idx - 1
            st.rerun()

    with col2:
        if st.button("➡️ Next", disabled=group_idx >= len(filtered) - 1, key=f"{filter_key}_next", use_container_width=True):
            st.session_state[nav_key] = group_idx + 1
            st.rerun()

    with col3:
        if st.button("💾 Save & Next", type="primary", key=f"{filter_key}_save", use_container_width=True):
            for idx, sel in selections.items():
                dkey = get_decision_key(sel["path"])
                progress["decisions"][dkey] = sel
            if group["hash"] not in progress["reviewed_hashes"]:
                progress["reviewed_hashes"].append(group["hash"])
            save_progress(progress)
            if group_idx < len(filtered) - 1:
                st.session_state[nav_key] = group_idx + 1
            st.rerun()

    with col4:
        if st.button("🔄 Relabel All", key=f"{filter_key}_relabel_all", use_container_width=True):
            for idx, sel in selections.items():
                sel["action"] = ACTION_RELABEL
                dkey = get_decision_key(sel["path"])
                progress["decisions"][dkey] = sel
            if group["hash"] not in progress["reviewed_hashes"]:
                progress["reviewed_hashes"].append(group["hash"])
            save_progress(progress)
            if group_idx < len(filtered) - 1:
                st.session_state[nav_key] = group_idx + 1
            st.rerun()


# ============================================================
# Main App
# ============================================================

def main():
    st.title("🏷️ Mislabel Image Reviewer")
    st.markdown(
        "Review images where multiple models disagree with the ground-truth label. "
        "For each image choose: **Keep** (label is correct), **Remove** (bad image), "
        "or **Relabel** (move to predicted class)."
    )

    # ---- Sidebar ----
    with st.sidebar:
        st.header("⚙️ Settings")

        tier1_path = st.text_input(
            "Tier 1 CSV (3/3 consensus)",
            value=DEFAULT_TIER1_CSV,
        )
        tier2_path = st.text_input(
            "Tier 2 CSV (2/3 consensus)",
            value=DEFAULT_TIER2_CSV,
        )

        st.markdown("---")
        if st.button("🔄 Reload CSVs", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # Load data
    df1, groups1 = load_mislabel_csv(tier1_path)
    df2, groups2 = load_mislabel_csv(tier2_path)

    # Initialize progress
    if "progress" not in st.session_state:
        st.session_state.progress = load_progress()
    progress = st.session_state.progress

    # ---- Sidebar stats ----
    with st.sidebar:
        st.header("📊 Statistics")

        n_tier1 = len(df1) if df1 is not None else 0
        n_tier2 = len(df2) if df2 is not None else 0
        n_groups1 = len(groups1)
        n_groups2 = len(groups2)

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Tier 1 images", n_tier1)
            st.metric("Tier 1 groups", n_groups1)
        with c2:
            st.metric("Tier 2 images", n_tier2)
            st.metric("Tier 2 groups", n_groups2)

        st.markdown("---")

        total_groups = n_groups1 + n_groups2
        reviewed = len(progress["reviewed_hashes"])
        n_decisions = len(progress["decisions"])

        st.metric("Groups reviewed", f"{reviewed} / {total_groups}")
        st.progress(reviewed / total_groups if total_groups > 0 else 0)

        # Decision breakdown
        actions = [d["action"] for d in progress["decisions"].values()]
        n_keep = actions.count(ACTION_KEEP)
        n_remove = actions.count(ACTION_REMOVE)
        n_relabel = actions.count(ACTION_RELABEL)

        st.markdown("**Decisions so far:**")
        st.write(f"- ✅ Keep: {n_keep}")
        st.write(f"- 🗑️ Remove: {n_remove}")
        st.write(f"- 🔄 Relabel: {n_relabel}")

        st.markdown("---")

        # Export
        st.header("💾 Export")
        if st.button("📥 Export decisions CSV", use_container_width=True):
            result = export_decisions(progress)
            if result is not None:
                st.success(f"Exported {len(result)} decisions to {EXPORT_PATH}")
            else:
                st.warning("No decisions to export yet.")

        if st.button("🗑️ Clear all progress", use_container_width=True):
            st.session_state.progress = {"decisions": {}, "reviewed_hashes": []}
            save_progress(st.session_state.progress)
            st.rerun()

    # ---- Main tabs ----
    tab1, tab2, tab3, tab4 = st.tabs([
        f"🔴 Tier 1 — 3/3 ({n_tier1})",
        f"🟡 Tier 2 — 2/3 ({n_tier2})",
        "📋 Decisions",
        "🚀 Apply",
    ])

    with tab1:
        render_tab("tier1", groups1, progress, "tier1_nav", "t1")

    with tab2:
        render_tab("tier2", groups2, progress, "tier2_nav", "t2")

    # ---- Tab 3: Decisions table ----
    with tab3:
        st.markdown("### All Decisions")
        if not progress["decisions"]:
            st.info("No decisions yet. Review groups in the Tier 1 / Tier 2 tabs.")
        else:
            dec_df = pd.DataFrame(progress["decisions"].values())

            # Summary
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Total", len(dec_df))
            with c2:
                st.metric("Keep", len(dec_df[dec_df["action"] == ACTION_KEEP]))
            with c3:
                st.metric("Remove", len(dec_df[dec_df["action"] == ACTION_REMOVE]))
            with c4:
                st.metric("Relabel", len(dec_df[dec_df["action"] == ACTION_RELABEL]))

            # Filter by action
            action_filter = st.selectbox(
                "Filter by action",
                ["All", ACTION_KEEP, ACTION_REMOVE, ACTION_RELABEL],
                format_func=lambda x: ACTION_LABELS.get(x, "All"),
                key="dec_filter",
            )
            display_df = dec_df if action_filter == "All" else dec_df[dec_df["action"] == action_filter]

            st.dataframe(
                display_df[["path", "original_label", "predicted_label", "action", "tier", "convnext_confidence"]],
                use_container_width=True,
                height=500,
            )

            csv_data = dec_df.to_csv(index=False)
            st.download_button(
                "📥 Download decisions CSV",
                csv_data,
                file_name=f"mislabel_decisions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # ---- Tab 4: Apply changes ----
    with tab4:
        st.markdown("### Apply Mislabel Corrections")
        st.markdown(
            "After reviewing, use this tab to generate a script that applies your decisions "
            "to the `foodsg-233_cleaned` dataset.\n\n"
            "- **Remove**: deletes the image file\n"
            "- **Relabel**: moves the image to the predicted class folder\n"
            "- **Keep**: no action needed"
        )

        if not progress["decisions"]:
            st.info("No decisions to apply yet.")
        else:
            dec_df = pd.DataFrame(progress["decisions"].values())
            removals = dec_df[dec_df["action"] == ACTION_REMOVE]
            relabels = dec_df[dec_df["action"] == ACTION_RELABEL]

            st.write(f"**{len(removals)}** images to remove, **{len(relabels)}** images to relabel")

            # Dry run preview
            with st.expander("Preview changes", expanded=True):
                if len(removals) > 0:
                    st.markdown("**Removals (first 20):**")
                    for _, row in removals.head(20).iterrows():
                        st.text(f"  DELETE {row['path']}")
                if len(relabels) > 0:
                    st.markdown("**Relabels (first 20):**")
                    for _, row in relabels.head(20).iterrows():
                        st.text(f"  MOVE {row['path']}  →  {row['predicted_label']}/")

            st.markdown("---")
            st.warning(
                "⚠️ This will modify your dataset on disk. Make sure you have a backup "
                "or that removals were already backed up by previous cleaning steps."
            )

            if st.button("🚀 Apply changes to dataset", type="primary", use_container_width=True):
                base_dir = Path(__file__).resolve().parent.parent  # food_classifier_project/
                errors = []
                applied_remove = 0
                applied_relabel = 0

                # Apply removals
                for _, row in removals.iterrows():
                    src = row["path"].replace("\\", "/")
                    if src.startswith("../"):
                        full_path = (base_dir / src).resolve()
                    else:
                        full_path = Path(src).resolve()
                    try:
                        if full_path.exists():
                            full_path.unlink()
                            applied_remove += 1
                    except Exception as e:
                        errors.append(f"Remove {src}: {e}")

                # Apply relabels
                for _, row in relabels.iterrows():
                    src = row["path"].replace("\\", "/")
                    pred = row["predicted_label"]
                    if src.startswith("../"):
                        full_path = (base_dir / src).resolve()
                    else:
                        full_path = Path(src).resolve()

                    # Determine target directory (same dataset root, different class folder)
                    # Path: ../foodsg-233_cleaned/OrigClass/image.jpg
                    # Target: ../foodsg-233_cleaned/PredClass/image.jpg
                    try:
                        parts = full_path.parts
                        # Find 'foodsg-233_cleaned' in path
                        root_idx = None
                        for pi, p in enumerate(parts):
                            if "foodsg-233_cleaned" in p:
                                root_idx = pi
                                break
                        if root_idx is None:
                            errors.append(f"Cannot find dataset root in {src}")
                            continue

                        target_dir = Path(*parts[: root_idx + 1]) / pred
                        target_dir.mkdir(parents=True, exist_ok=True)
                        target_path = target_dir / full_path.name

                        if full_path.exists():
                            shutil.move(str(full_path), str(target_path))
                            applied_relabel += 1
                    except Exception as e:
                        errors.append(f"Relabel {src}: {e}")

                st.success(
                    f"Applied: {applied_remove} removals, {applied_relabel} relabels"
                )
                if errors:
                    st.error(f"{len(errors)} errors:")
                    for err in errors[:20]:
                        st.text(err)


if __name__ == "__main__":
    main()
