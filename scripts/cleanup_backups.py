#!/usr/bin/env python3
"""
Backup Cleanup Script - Keeps only the most recent backups to prevent repo bloat
"""
import os
import shutil
from pathlib import Path
from datetime import datetime
import argparse

def cleanup_backups(backup_dir=".kiro/backups", keep_count=2, dry_run=False):
    """
    Clean up old backup directories, keeping only the most recent ones
    
    Args:
        backup_dir: Directory containing backups
        keep_count: Number of recent backups to keep
        dry_run: If True, only show what would be deleted
    """
    backup_path = Path(backup_dir)
    
    if not backup_path.exists():
        print(f"📁 Backup directory {backup_dir} doesn't exist")
        return
    
    # Find all backup directories
    backup_dirs = []
    for item in backup_path.iterdir():
        if item.is_dir() and item.name.startswith("optimization_backup_"):
            try:
                # Extract timestamp from directory name
                timestamp_str = item.name.replace("optimization_backup_", "")
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                backup_dirs.append((timestamp, item))
            except ValueError:
                print(f"⚠️  Skipping invalid backup directory: {item.name}")
    
    # Sort by timestamp (newest first)
    backup_dirs.sort(key=lambda x: x[0], reverse=True)
    
    print(f"📊 Found {len(backup_dirs)} backup directories")
    
    if len(backup_dirs) <= keep_count:
        print(f"✅ Only {len(backup_dirs)} backups found, keeping all (limit: {keep_count})")
        return
    
    # Keep the most recent ones, delete the rest
    to_keep = backup_dirs[:keep_count]
    to_delete = backup_dirs[keep_count:]
    
    print(f"📦 Keeping {len(to_keep)} most recent backups:")
    for timestamp, path in to_keep:
        size_mb = get_dir_size(path) / (1024 * 1024)
        print(f"  ✅ {path.name} ({timestamp.strftime('%Y-%m-%d %H:%M:%S')}) - {size_mb:.1f}MB")
    
    print(f"🗑️  {'Would delete' if dry_run else 'Deleting'} {len(to_delete)} old backups:")
    total_freed_mb = 0
    
    for timestamp, path in to_delete:
        size_mb = get_dir_size(path) / (1024 * 1024)
        total_freed_mb += size_mb
        
        print(f"  {'🔍' if dry_run else '❌'} {path.name} ({timestamp.strftime('%Y-%m-%d %H:%M:%S')}) - {size_mb:.1f}MB")
        
        if not dry_run:
            try:
                shutil.rmtree(path)
                print(f"    ✅ Deleted successfully")
            except Exception as e:
                print(f"    ❌ Failed to delete: {e}")
    
    print(f"💾 {'Would free' if dry_run else 'Freed'} {total_freed_mb:.1f}MB of disk space")

def get_dir_size(path):
    """Get total size of directory in bytes"""
    total = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total += os.path.getsize(filepath)
    except (OSError, IOError):
        pass
    return total

def main():
    parser = argparse.ArgumentParser(description="Clean up old backup directories")
    parser.add_argument("--backup-dir", default=".kiro/backups", 
                       help="Backup directory path (default: .kiro/backups)")
    parser.add_argument("--keep", type=int, default=2,
                       help="Number of recent backups to keep (default: 2)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be deleted without actually deleting")
    
    args = parser.parse_args()
    
    print("🧹 BACKUP CLEANUP")
    print("=" * 40)
    print(f"Backup Directory: {args.backup_dir}")
    print(f"Keep Count: {args.keep}")
    print(f"Dry Run: {args.dry_run}")
    print("=" * 40)
    
    cleanup_backups(args.backup_dir, args.keep, args.dry_run)

if __name__ == "__main__":
    main()