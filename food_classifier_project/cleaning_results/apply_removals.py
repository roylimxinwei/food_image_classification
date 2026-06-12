"""
Apply Removals Script
=====================
After reviewing duplicates in the Streamlit app, run this script to:
1. Move images to a backup folder (safe)
2. Or permanently delete them (careful!)

Usage:
    python apply_removals.py                    # Move to backup (default)
    python apply_removals.py --delete           # Permanently delete
    python apply_removals.py --dry-run          # Preview only
"""

import argparse
import pandas as pd
import os
import shutil
from pathlib import Path
from datetime import datetime


def apply_removals(
    csv_path: str = "images_to_remove.csv",
    backup_dir: str = "removed_images_backup",
    delete: bool = False,
    dry_run: bool = False
):
    """
    Apply removals from the review CSV.
    
    Args:
        csv_path: Path to the removal list CSV
        backup_dir: Where to move files (if not deleting)
        delete: If True, permanently delete. If False, move to backup.
        dry_run: If True, only print what would happen
    """
    
    # Load removal list
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        print("   Run the Streamlit reviewer first and export the removal list.")
        return
    
    df = pd.read_csv(csv_path)
    print(f"📋 Loaded {len(df)} images to remove")
    
    if len(df) == 0:
        print("   Nothing to remove!")
        return
    
    # Summary by reason
    print("\n📊 Summary:")
    for reason, count in df['reason'].value_counts().items():
        print(f"   • {reason}: {count}")
    
    # Confirm action
    action = "DELETE" if delete else "MOVE to backup"
    print(f"\n⚠️  Action: {action}")
    
    if dry_run:
        print("\n🔍 DRY RUN - No files will be modified\n")
    else:
        if delete:
            confirm = input("\n⚠️  This will PERMANENTLY DELETE files. Type 'DELETE' to confirm: ")
            if confirm != "DELETE":
                print("❌ Cancelled")
                return
        else:
            confirm = input(f"\nMove files to '{backup_dir}'? [y/N]: ")
            if confirm.lower() != 'y':
                print("❌ Cancelled")
                return
    
    # Create backup directory
    if not delete and not dry_run:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = Path(backup_dir) / timestamp
        backup_path.mkdir(parents=True, exist_ok=True)
        print(f"\n📁 Backup directory: {backup_path}")
    
    # Process files
    success = 0
    failed = 0
    not_found = 0
    
    for _, row in df.iterrows():
        path = row['path'].replace('\\', '/')
        
        if not os.path.exists(path):
            print(f"   ⚠️  Not found: {path}")
            not_found += 1
            continue
        
        try:
            if dry_run:
                print(f"   Would {action.lower()}: {path}")
            elif delete:
                os.remove(path)
                print(f"   🗑️  Deleted: {path}")
            else:
                # Move to backup, preserving label folder structure
                label = row.get('label', 'unknown')
                dest_dir = backup_path / label
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_path = dest_dir / Path(path).name
                shutil.move(path, dest_path)
                print(f"   📦 Moved: {Path(path).name} → {dest_dir.name}/")
            success += 1
        except Exception as e:
            print(f"   ❌ Failed: {path} - {e}")
            failed += 1
    
    # Summary
    print(f"\n✅ Complete!")
    print(f"   • Success: {success}")
    print(f"   • Failed: {failed}")
    print(f"   • Not found: {not_found}")
    
    if not delete and not dry_run and success > 0:
        print(f"\n💡 Files backed up to: {backup_path}")
        print("   You can delete this folder once you're sure everything is correct.")


def main():
    parser = argparse.ArgumentParser(description="Apply image removals from review")
    parser.add_argument(
        "--csv",
        default="images_to_remove.csv",
        help="Path to removal list CSV (default: images_to_remove.csv)"
    )
    parser.add_argument(
        "--backup-dir",
        default="removed_images_backup",
        help="Backup directory for moved files (default: removed_images_backup)"
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Permanently delete files instead of moving to backup"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without modifying files"
    )
    
    args = parser.parse_args()
    
    apply_removals(
        csv_path=args.csv,
        backup_dir=args.backup_dir,
        delete=args.delete,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
