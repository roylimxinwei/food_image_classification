"""
Duplicate Image Reviewer - Streamlit App
=========================================
Review duplicate images side by side and mark which ones to remove.

Usage:
    streamlit run duplicate_reviewer.py

Features:
- View exact and near duplicates grouped by hash
- See images side by side with their labels
- Select images to remove with checkboxes
- Export removal list to CSV
- Track progress through duplicate groups
"""

import streamlit as st
import pandas as pd
from PIL import Image
from pathlib import Path
import os
import json
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Duplicate Image Reviewer",
    page_icon="🔍",
    layout="wide"
)

# ============================================================
# Configuration
# ============================================================

# Update these paths to match your setup
DEFAULT_FOODSG233_PATH = "../../foodsg-233"  # Path to the foodsg-233 dataset to be cleaned
DEFAULT_EXACT_DUPES_PATH = "02a_exact_duplicates.csv"
DEFAULT_NEAR_DUPES_PATH = "02b_near_duplicates.csv"
OUTPUT_PATH = "images_to_remove.csv"
PROGRESS_PATH = "review_progress.json"


# ============================================================
# Helper Functions
# ============================================================

@st.cache_data
def load_duplicates(exact_path: str, near_path: str):
    """Load duplicate CSVs and group by hash."""
    data = {
        'exact': {'df': None, 'groups': [], 'hash_col': 'md5'},
        'near': {'df': None, 'groups': [], 'hash_col': 'phash'},
    }
    
    # Load exact duplicates
    if os.path.exists(exact_path):
        df = pd.read_csv(exact_path)
        data['exact']['df'] = df
        # Group by md5 hash
        groups = []
        for hash_val, group in df.groupby('md5'):
            if len(group) > 1:  # Only include actual duplicates
                groups.append({
                    'hash': hash_val,
                    'images': group.to_dict('records'),
                    'labels': list(group['label'].unique()),
                    'is_cross_class': group['label'].nunique() > 1
                })
        # Sort: cross-class first, then by number of images
        groups.sort(key=lambda x: (-x['is_cross_class'], -len(x['images'])))
        data['exact']['groups'] = groups
    
    # Load near duplicates
    if os.path.exists(near_path):
        df = pd.read_csv(near_path)
        data['near']['df'] = df
        # Group by phash
        groups = []
        for hash_val, group in df.groupby('phash'):
            if len(group) > 1:
                groups.append({
                    'hash': hash_val,
                    'images': group.to_dict('records'),
                    'labels': list(group['label'].unique()),
                    'is_cross_class': group['label'].nunique() > 1
                })
        groups.sort(key=lambda x: (-x['is_cross_class'], -len(x['images'])))
        data['near']['groups'] = groups
    
    return data


def get_cleaned_path(original_path: str) -> str:
    """Convert original dataset path to cleaned dataset path."""
    clean_path = original_path.replace('\\', '/')
    # Replace original dataset path with cleaned dataset path
    clean_path = clean_path.replace('../foodsg-233/', DEFAULT_FOODSG233_PATH + '/')
    return clean_path


def load_image_safe(path: str, max_size: int = 400):
    """Load image with error handling and resize."""
    try:
        # Handle different path formats
        clean_path = path.replace('\\', '/')
        
        # If path is relative (starts with ../), resolve it from parent directory
        # since CSV paths are relative to food_classifier_project/ but app runs from cleaning_results/
        if clean_path.startswith('../'):
            # Go up one level from current directory (cleaning_results -> food_classifier_project)
            base_dir = Path(__file__).parent.parent
            # Use the cleaned dataset instead of original
            clean_path = get_cleaned_path(clean_path)
            clean_path = base_dir / clean_path
        
        img = Image.open(clean_path)
        img = img.convert('RGB')
        
        # Resize for display
        ratio = max_size / max(img.size)
        if ratio < 1:
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        return img
    except Exception as e:
        return None


def load_progress():
    """Load review progress from file."""
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, 'r') as f:
            return json.load(f)
    return {'reviewed_hashes': [], 'to_remove': []}


def save_progress(progress):
    """Save review progress to file."""
    with open(PROGRESS_PATH, 'w') as f:
        json.dump(progress, f, indent=2)


def export_removal_list(to_remove: list):
    """Export list of images to remove to CSV."""
    if to_remove:
        df = pd.DataFrame(to_remove)
        df.to_csv(OUTPUT_PATH, index=False)
        return True
    return False


# ============================================================
# Main App
# ============================================================

def main():
    st.title("🔍 Duplicate Image Reviewer")
    st.markdown("Review duplicate images and select which ones to remove.")
    
    # Sidebar - File paths and settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        exact_path = st.text_input(
            "Exact duplicates CSV",
            value=DEFAULT_EXACT_DUPES_PATH,
            help="Path to 02a_exact_duplicates.csv"
        )
        
        near_path = st.text_input(
            "Near duplicates CSV",
            value=DEFAULT_NEAR_DUPES_PATH,
            help="Path to 02b_near_duplicates.csv"
        )
        
        st.markdown("---")
        
        # Base path for images (to fix relative paths)
        st.subheader("🖼️ Image Path Fix")
        base_path = st.text_input(
            "Dataset base path",
            value="",
            help="If images don't load, enter the full path to your dataset folder. "
                 "E.g., C:/Users/you/data or /home/you/data"
        )
        
        path_prefix_to_remove = st.text_input(
            "Path prefix to remove",
            value="../",
            help="Remove this prefix from CSV paths before adding base path"
        )
        
        # Store in session state
        st.session_state['base_path'] = base_path
        st.session_state['path_prefix_to_remove'] = path_prefix_to_remove
        
        # Debug: show example path transformation
        if base_path:
            example_original = "../foodsg-233/example/image.jpg"
            example_clean = example_original
            if path_prefix_to_remove and example_clean.startswith(path_prefix_to_remove):
                example_clean = example_clean[len(path_prefix_to_remove):]
            example_final = base_path.rstrip('/\\') + '/' + example_clean
            st.caption(f"Example path:\n`{example_original}`\n→ `{example_final}`")
        
        st.markdown("---")
        
        # Load data
        if st.button("🔄 Reload Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # Load data
    try:
        data = load_duplicates(exact_path, near_path)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Please check that the CSV files exist and have the correct format.")
        return
    
    # Initialize session state
    if 'progress' not in st.session_state:
        st.session_state.progress = load_progress()
    
    if 'current_type' not in st.session_state:
        st.session_state.current_type = 'exact'
    
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    
    # Sidebar - Statistics
    with st.sidebar:
        st.header("📊 Statistics")
        
        exact_groups = data['exact']['groups']
        near_groups = data['near']['groups']
        
        exact_cross = sum(1 for g in exact_groups if g['is_cross_class'])
        near_cross = sum(1 for g in near_groups if g['is_cross_class'])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Exact Groups", len(exact_groups))
            st.metric("Cross-class", exact_cross, delta="⚠️" if exact_cross > 0 else None)
        with col2:
            st.metric("Near Groups", len(near_groups))
            st.metric("Cross-class", near_cross, delta="⚠️" if near_cross > 0 else None)
        
        st.markdown("---")
        
        # Progress
        reviewed = len(st.session_state.progress['reviewed_hashes'])
        total = len(exact_groups) + len(near_groups)
        to_remove = len(st.session_state.progress['to_remove'])
        
        st.metric("Reviewed", f"{reviewed}/{total}")
        st.metric("Marked for Removal", to_remove)
        
        progress_pct = reviewed / total if total > 0 else 0
        st.progress(progress_pct)
        
        st.markdown("---")
        
        # Export
        st.header("💾 Export")
        if st.button("📥 Export Removal List", use_container_width=True):
            if export_removal_list(st.session_state.progress['to_remove']):
                st.success(f"Exported {to_remove} images to {OUTPUT_PATH}")
            else:
                st.warning("No images marked for removal yet.")
        
        if st.button("🗑️ Clear Progress", use_container_width=True):
            st.session_state.progress = {'reviewed_hashes': [], 'to_remove': []}
            save_progress(st.session_state.progress)
            st.rerun()
    
    # Main content - Tabs for exact vs near
    tab1, tab2, tab3 = st.tabs(["📋 Exact Duplicates", "🔄 Near Duplicates", "📄 Removal List"])
    
    # --------------------------------------------------------
    # Tab 1: Exact Duplicates
    # --------------------------------------------------------
    with tab1:
        groups = data['exact']['groups']
        
        if not groups:
            st.info("No exact duplicates found.")
        else:
            # Filter options
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                filter_option = st.selectbox(
                    "Filter",
                    ["All", "Cross-class only", "Same-class only", "Not reviewed"],
                    key="exact_filter"
                )
            with col2:
                # Apply filter
                filtered_groups = groups.copy()
                if filter_option == "Cross-class only":
                    filtered_groups = [g for g in groups if g['is_cross_class']]
                elif filter_option == "Same-class only":
                    filtered_groups = [g for g in groups if not g['is_cross_class']]
                elif filter_option == "Not reviewed":
                    filtered_groups = [g for g in groups if g['hash'] not in st.session_state.progress['reviewed_hashes']]
                
                st.write(f"Showing {len(filtered_groups)} groups")
            
            if filtered_groups:
                # Navigation - initialize session state if needed
                if 'exact_nav' not in st.session_state:
                    st.session_state['exact_nav'] = 0
                
                # Clamp to valid range
                st.session_state['exact_nav'] = max(0, min(st.session_state['exact_nav'], len(filtered_groups) - 1))
                
                with col3:
                    group_idx = st.number_input(
                        "Group #",
                        min_value=1,
                        max_value=len(filtered_groups),
                        value=st.session_state['exact_nav'] + 1,
                        key="exact_idx"
                    ) - 1
                    # Sync back to session state
                    st.session_state['exact_nav'] = group_idx
                
                # Display current group
                group = filtered_groups[group_idx]
                
                # Header
                is_reviewed = group['hash'] in st.session_state.progress['reviewed_hashes']
                status = "✅ Reviewed" if is_reviewed else "⏳ Pending"
                cross_badge = "🔴 CROSS-CLASS" if group['is_cross_class'] else "🟢 Same class"
                
                st.markdown(f"### Group {group_idx + 1}/{len(filtered_groups)} {cross_badge} {status}")
                st.markdown(f"**Hash:** `{group['hash'][:16]}...` | **Labels:** {', '.join(group['labels'])}")
                
                # Display images
                images = group['images']
                n_cols = min(len(images), 4)
                cols = st.columns(n_cols)
                
                selections = {}
                for i, img_info in enumerate(images):
                    col_idx = i % n_cols
                    with cols[col_idx]:
                        # Load and display image
                        img = load_image_safe(img_info['path'])
                        if img:
                            st.image(img, use_container_width=True)
                        else:
                            st.error("❌ Cannot load image")
                            # Show debug info
                            with st.expander("Debug path info"):
                                original = img_info['path']
                                clean = original.replace('\\', '/')
                                prefix = st.session_state.get('path_prefix_to_remove', '')
                                bp = st.session_state.get('base_path', '')
                                if prefix and clean.startswith(prefix):
                                    clean = clean[len(prefix):]
                                if bp:
                                    final = bp.rstrip('/\\') + '/' + clean
                                else:
                                    final = clean
                                st.code(f"Original: {original}\nFinal: {final}")
                        
                        # Image info
                        st.markdown(f"**Label:** {img_info['label']}")
                        st.caption(f"Split: {img_info['split']}")
                        st.caption(Path(img_info['path']).name[:30] + "...")
                        
                        # Checkbox to mark for removal
                        # Check if already marked
                        is_marked = any(
                            r['path'] == get_cleaned_path(img_info['path']) 
                            for r in st.session_state.progress['to_remove']
                        )
                        
                        key = f"exact_{group['hash']}_{i}"
                        remove = st.checkbox(
                            "🗑️ Remove",
                            value=is_marked,
                            key=key
                        )
                        selections[i] = remove
                
                # Action buttons
                st.markdown("---")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if st.button("⬅️ Previous", use_container_width=True, disabled=group_idx == 0):
                        st.session_state['exact_nav'] = group_idx - 1
                        st.rerun()
                
                with col2:
                    if st.button("➡️ Next", use_container_width=True, disabled=group_idx >= len(filtered_groups) - 1):
                        st.session_state['exact_nav'] = group_idx + 1
                        st.rerun()
                
                with col3:
                    if st.button("💾 Save Selections", type="primary", use_container_width=True):
                        # Update removal list
                        for i, remove in selections.items():
                            img_info = images[i]
                            # Remove from list first (to avoid duplicates)
                            cleaned_path = get_cleaned_path(img_info['path'])
                            st.session_state.progress['to_remove'] = [
                                r for r in st.session_state.progress['to_remove']
                                if r['path'] != cleaned_path
                            ]
                            # Add if marked for removal
                            if remove:
                                st.session_state.progress['to_remove'].append({
                                    'path': cleaned_path,
                                    'label': img_info['label'],
                                    'split': img_info['split'],
                                    'reason': 'exact_duplicate',
                                    'hash': group['hash']
                                })
                        
                        # Mark as reviewed
                        if group['hash'] not in st.session_state.progress['reviewed_hashes']:
                            st.session_state.progress['reviewed_hashes'].append(group['hash'])
                        
                        save_progress(st.session_state.progress)
                        
                        # Auto-advance to next group
                        if group_idx < len(filtered_groups) - 1:
                            st.session_state['exact_nav'] = group_idx + 1
                        
                        st.success("✅ Saved!")
                        st.rerun()
                
                with col4:
                    if st.button("🗑️ Remove All But First", use_container_width=True):
                        # Mark all except first for removal
                        for i, img_info in enumerate(images):
                            cleaned_path = get_cleaned_path(img_info['path'])
                            st.session_state.progress['to_remove'] = [
                                r for r in st.session_state.progress['to_remove']
                                if r['path'] != cleaned_path
                            ]
                            if i > 0:  # Keep first, remove rest
                                st.session_state.progress['to_remove'].append({
                                    'path': cleaned_path,
                                    'label': img_info['label'],
                                    'split': img_info['split'],
                                    'reason': 'exact_duplicate',
                                    'hash': group['hash']
                                })
                        
                        if group['hash'] not in st.session_state.progress['reviewed_hashes']:
                            st.session_state.progress['reviewed_hashes'].append(group['hash'])
                        
                        save_progress(st.session_state.progress)
                        
                        # Auto-advance to next group
                        if group_idx < len(filtered_groups) - 1:
                            st.session_state['exact_nav'] = group_idx + 1
                        
                        st.success("✅ Marked all but first for removal!")
                        st.rerun()
    
    # --------------------------------------------------------
    # Tab 2: Near Duplicates
    # --------------------------------------------------------
    with tab2:
        groups = data['near']['groups']
        
        if not groups:
            st.info("No near duplicates found.")
        else:
            # Filter options
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                filter_option = st.selectbox(
                    "Filter",
                    ["All", "Cross-class only", "Same-class only", "Not reviewed"],
                    key="near_filter"
                )
            with col2:
                filtered_groups = groups.copy()
                if filter_option == "Cross-class only":
                    filtered_groups = [g for g in groups if g['is_cross_class']]
                elif filter_option == "Same-class only":
                    filtered_groups = [g for g in groups if not g['is_cross_class']]
                elif filter_option == "Not reviewed":
                    filtered_groups = [g for g in groups if g['hash'] not in st.session_state.progress['reviewed_hashes']]
                
                st.write(f"Showing {len(filtered_groups)} groups")
            
            if filtered_groups:
                # Navigation - initialize session state if needed
                if 'near_nav' not in st.session_state:
                    st.session_state['near_nav'] = 0
                
                # Clamp to valid range
                st.session_state['near_nav'] = max(0, min(st.session_state['near_nav'], len(filtered_groups) - 1))
                
                with col3:
                    group_idx = st.number_input(
                        "Group #",
                        min_value=1,
                        max_value=len(filtered_groups),
                        value=st.session_state['near_nav'] + 1,
                        key="near_idx"
                    ) - 1
                    # Sync back to session state
                    st.session_state['near_nav'] = group_idx
                
                group = filtered_groups[group_idx]
                
                is_reviewed = group['hash'] in st.session_state.progress['reviewed_hashes']
                status = "✅ Reviewed" if is_reviewed else "⏳ Pending"
                cross_badge = "🔴 CROSS-CLASS" if group['is_cross_class'] else "🟢 Same class"
                
                st.markdown(f"### Group {group_idx + 1}/{len(filtered_groups)} {cross_badge} {status}")
                st.markdown(f"**pHash:** `{group['hash']}` | **Labels:** {', '.join(group['labels'])}")
                
                images = group['images']
                n_cols = min(len(images), 4)
                cols = st.columns(n_cols)
                
                selections = {}
                for i, img_info in enumerate(images):
                    col_idx = i % n_cols
                    with cols[col_idx]:
                        img = load_image_safe(img_info['path'])
                        if img:
                            st.image(img, use_container_width=True)
                        else:
                            st.error("❌ Cannot load image")
                            with st.expander("Debug"):
                                original = img_info['path']
                                clean = original.replace('\\', '/')
                                prefix = st.session_state.get('path_prefix_to_remove', '')
                                bp = st.session_state.get('base_path', '')
                                if prefix and clean.startswith(prefix):
                                    clean = clean[len(prefix):]
                                if bp:
                                    final = bp.rstrip('/\\') + '/' + clean
                                else:
                                    final = clean
                                st.code(f"Final: {final}")
                        
                        st.markdown(f"**Label:** {img_info['label']}")
                        st.caption(f"Split: {img_info['split']}")
                        st.caption(Path(img_info['path']).name[:30] + "...")
                        
                        is_marked = any(
                            r['path'] == get_cleaned_path(img_info['path'])
                            for r in st.session_state.progress['to_remove']
                        )
                        
                        key = f"near_{group['hash']}_{i}"
                        remove = st.checkbox(
                            "🗑️ Remove",
                            value=is_marked,
                            key=key
                        )
                        selections[i] = remove
                
                st.markdown("---")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if st.button("⬅️ Prev", use_container_width=True, disabled=group_idx == 0, key="near_prev"):
                        st.session_state['near_nav'] = group_idx - 1
                        st.rerun()
                
                with col2:
                    if st.button("➡️ Next", use_container_width=True, disabled=group_idx >= len(filtered_groups) - 1, key="near_next"):
                        st.session_state['near_nav'] = group_idx + 1
                        st.rerun()
                
                with col3:
                    if st.button("💾 Save", type="primary", use_container_width=True, key="near_save"):
                        for i, remove in selections.items():
                            img_info = images[i]
                            cleaned_path = get_cleaned_path(img_info['path'])
                            st.session_state.progress['to_remove'] = [
                                r for r in st.session_state.progress['to_remove']
                                if r['path'] != cleaned_path
                            ]
                            if remove:
                                st.session_state.progress['to_remove'].append({
                                    'path': cleaned_path,
                                    'label': img_info['label'],
                                    'split': img_info['split'],
                                    'reason': 'near_duplicate',
                                    'hash': group['hash']
                                })
                        
                        if group['hash'] not in st.session_state.progress['reviewed_hashes']:
                            st.session_state.progress['reviewed_hashes'].append(group['hash'])
                        
                        save_progress(st.session_state.progress)
                        
                        # Auto-advance to next group
                        if group_idx < len(filtered_groups) - 1:
                            st.session_state['near_nav'] = group_idx + 1
                        
                        st.success("✅ Saved!")
                        st.rerun()
                
                with col4:
                    if st.button("🗑️ Remove All But First", use_container_width=True, key="near_remove_all"):
                        # Mark all except first for removal
                        for i, img_info in enumerate(images):
                            cleaned_path = get_cleaned_path(img_info['path'])
                            st.session_state.progress['to_remove'] = [
                                r for r in st.session_state.progress['to_remove']
                                if r['path'] != cleaned_path
                            ]
                            if i > 0:  # Keep first, remove rest
                                st.session_state.progress['to_remove'].append({
                                    'path': cleaned_path,
                                    'label': img_info['label'],
                                    'split': img_info['split'],
                                    'reason': 'near_duplicate',
                                    'hash': group['hash']
                                })
                        
                        if group['hash'] not in st.session_state.progress['reviewed_hashes']:
                            st.session_state.progress['reviewed_hashes'].append(group['hash'])
                        
                        save_progress(st.session_state.progress)
                        
                        # Auto-advance to next group
                        if group_idx < len(filtered_groups) - 1:
                            st.session_state['near_nav'] = group_idx + 1
                        
                        st.success("✅ Marked all but first for removal!")
                        st.rerun()
    
    # --------------------------------------------------------
    # Tab 3: Removal List
    # --------------------------------------------------------
    with tab3:
        st.markdown("### Images Marked for Removal")
        
        to_remove = st.session_state.progress['to_remove']
        
        if not to_remove:
            st.info("No images marked for removal yet.")
        else:
            # Summary
            df = pd.DataFrame(to_remove)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total to Remove", len(df))
            with col2:
                st.metric("Exact Duplicates", len(df[df['reason'] == 'exact_duplicate']))
            with col3:
                st.metric("Near Duplicates", len(df[df['reason'] == 'near_duplicate']))
            
            # Table
            st.dataframe(
                df[['path', 'label', 'reason']],
                use_container_width=True,
                height=400
            )
            
            # Export
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv,
                    file_name=f"images_to_remove_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                if st.button("🗑️ Clear All Selections", use_container_width=True):
                    st.session_state.progress['to_remove'] = []
                    save_progress(st.session_state.progress)
                    st.rerun()
            
            # Preview some images
            st.markdown("### Preview")
            preview_cols = st.columns(5)
            for i, (_, row) in enumerate(df.head(5).iterrows()):
                with preview_cols[i]:
                    img = load_image_safe(row['path'], max_size=150)
                    if img:
                        st.image(img, caption=row['label'][:15], use_container_width=True)


if __name__ == "__main__":
    main()
