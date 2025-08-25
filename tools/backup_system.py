#!/usr/bin/env python3
"""
Pre-optimization backup system for codebase optimization.
Creates backups of critical files and provides rollback mechanism.
"""

import os
import shutil
import json
import datetime
from pathlib import Path
from typing import Dict, List, Optional
import hashlib

class BackupSystem:
    """Handles backup and rollback operations for codebase optimization."""
    
    def __init__(self, backup_dir: str = ".kiro/backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_backup_dir = self.backup_dir / f"optimization_backup_{self.timestamp}"
        self.manifest_file = self.current_backup_dir / "backup_manifest.json"
        self.manifest = {
            "timestamp": self.timestamp,
            "backup_type": "codebase_optimization",
            "files": {},
            "structure": {},
            "checksums": {}
        }
    
    def create_backup(self, critical_files: Optional[List[str]] = None) -> str:
        """Create backup of critical files and document structure."""
        if critical_files is None:
            critical_files = self._identify_critical_files()
        
        self.current_backup_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🔄 Creating backup in {self.current_backup_dir}")
        
        # Document current file structure
        self._document_structure()
        
        # Backup critical files
        for file_path in critical_files:
            self._backup_file(file_path)
        
        # Save manifest
        self._save_manifest()
        
        print(f"✅ Backup created successfully: {len(critical_files)} files backed up")
        return str(self.current_backup_dir)
    
    def _identify_critical_files(self) -> List[str]:
        """Identify critical files that need backup."""
        critical_files = []
        
        # Test runners and related files
        test_patterns = [
            "test_runner_fast.py",
            "run_tests.py", 
            "test_runner_comprehensive.py",
            "run_fast_tests.py",
            "run_minimal_tests.py",
            "run_minimal_fast_tests.py",
            "run_optimized_complete_tests.py",
            "run_complete_tests_with_analysis.py"
        ]
        
        # Analysis and summary files
        analysis_patterns = [
            "COMPLETE_TEST_ANALYSIS_SUMMARY.md",
            "FINAL_SPEED_SOLUTION.md",
            "LOCKED_IN_SPEED_SOLUTION.md",
            "SPEED_FIX_SUMMARY.md",
            "TEST_SUITE_LOCKED_IN.md"
        ]
        
        # JSON analysis files
        json_patterns = [
            "*.json"
        ]
        
        # Check for existing files
        for pattern in test_patterns + analysis_patterns:
            if Path(pattern).exists():
                critical_files.append(pattern)
        
        # Find JSON analysis files
        for json_file in Path(".").glob("ultra_fast_test_analysis_*.json"):
            critical_files.append(str(json_file))
        
        for json_file in Path(".").glob("*analysis*.json"):
            critical_files.append(str(json_file))
        
        # Important test files that might be renamed
        test_files = [
            "tests/unit/test_markdown_chunking.py",
            "tests/unit/test_sota_markdown_chunking.py"
        ]
        
        for test_file in test_files:
            if Path(test_file).exists():
                critical_files.append(test_file)
        
        return list(set(critical_files))  # Remove duplicates
    
    def _backup_file(self, file_path: str) -> None:
        """Backup a single file with checksum verification."""
        source_path = Path(file_path)
        if not source_path.exists():
            print(f"⚠️  Warning: {file_path} not found, skipping")
            return
        
        # Create backup path maintaining directory structure
        relative_path = source_path.relative_to(Path("."))
        backup_path = self.current_backup_dir / "files" / relative_path
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file
        shutil.copy2(source_path, backup_path)
        
        # Calculate checksum
        checksum = self._calculate_checksum(source_path)
        
        # Record in manifest
        self.manifest["files"][str(relative_path)] = {
            "original_path": str(source_path),
            "backup_path": str(backup_path),
            "size": source_path.stat().st_size,
            "modified": source_path.stat().st_mtime,
            "checksum": checksum
        }
        
        print(f"  📁 Backed up: {file_path}")
    
    def _document_structure(self) -> None:
        """Document current file structure for reference."""
        structure = {}
        
        # Document key directories
        key_dirs = [".", "tests", "services", "tools", "utils", "models"]
        
        for dir_path in key_dirs:
            if Path(dir_path).exists():
                structure[dir_path] = self._get_directory_listing(dir_path)
        
        self.manifest["structure"] = structure
        
        # Save detailed structure to separate file
        structure_file = self.current_backup_dir / "original_structure.json"
        with open(structure_file, 'w') as f:
            json.dump(structure, f, indent=2)
    
    def _get_directory_listing(self, dir_path: str) -> Dict:
        """Get detailed directory listing."""
        listing = {"files": [], "directories": []}
        
        try:
            for item in Path(dir_path).iterdir():
                if item.is_file():
                    listing["files"].append({
                        "name": item.name,
                        "size": item.stat().st_size,
                        "modified": item.stat().st_mtime
                    })
                elif item.is_dir() and not item.name.startswith('.'):
                    listing["directories"].append(item.name)
        except PermissionError:
            listing["error"] = "Permission denied"
        
        return listing
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _save_manifest(self) -> None:
        """Save backup manifest."""
        with open(self.manifest_file, 'w') as f:
            json.dump(self.manifest, f, indent=2)
    
    def rollback(self, backup_path: str) -> bool:
        """Rollback changes using specified backup."""
        backup_dir = Path(backup_path)
        manifest_file = backup_dir / "backup_manifest.json"
        
        if not manifest_file.exists():
            print(f"❌ Backup manifest not found: {manifest_file}")
            return False
        
        # Load manifest
        with open(manifest_file, 'r') as f:
            manifest = json.load(f)
        
        print(f"🔄 Rolling back using backup from {manifest['timestamp']}")
        
        # Restore files
        success_count = 0
        for relative_path, file_info in manifest["files"].items():
            if self._restore_file(backup_dir, relative_path, file_info):
                success_count += 1
        
        print(f"✅ Rollback completed: {success_count}/{len(manifest['files'])} files restored")
        return success_count == len(manifest["files"])
    
    def _restore_file(self, backup_dir: Path, relative_path: str, file_info: Dict) -> bool:
        """Restore a single file from backup."""
        backup_file_path = backup_dir / "files" / relative_path
        original_path = Path(file_info["original_path"])
        
        if not backup_file_path.exists():
            print(f"❌ Backup file not found: {backup_file_path}")
            return False
        
        try:
            # Create parent directories if needed
            original_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Restore file
            shutil.copy2(backup_file_path, original_path)
            
            # Verify checksum
            if self._calculate_checksum(original_path) == file_info["checksum"]:
                print(f"  ✅ Restored: {relative_path}")
                return True
            else:
                print(f"  ⚠️  Checksum mismatch: {relative_path}")
                return False
                
        except Exception as e:
            print(f"  ❌ Failed to restore {relative_path}: {e}")
            return False
    
    def list_backups(self) -> List[Dict]:
        """List available backups."""
        backups = []
        
        if not self.backup_dir.exists():
            return backups
        
        for backup_path in self.backup_dir.glob("optimization_backup_*"):
            manifest_file = backup_path / "backup_manifest.json"
            if manifest_file.exists():
                try:
                    with open(manifest_file, 'r') as f:
                        manifest = json.load(f)
                    
                    backups.append({
                        "path": str(backup_path),
                        "timestamp": manifest["timestamp"],
                        "file_count": len(manifest["files"]),
                        "backup_type": manifest.get("backup_type", "unknown")
                    })
                except Exception as e:
                    print(f"⚠️  Error reading backup manifest {manifest_file}: {e}")
        
        return sorted(backups, key=lambda x: x["timestamp"], reverse=True)


def main():
    """CLI interface for backup system."""
    import sys
    
    backup_system = BackupSystem()
    
    if len(sys.argv) < 2:
        print("Usage: python backup_system.py <command> [args]")
        print("Commands:")
        print("  create - Create backup of critical files")
        print("  list - List available backups")
        print("  rollback <backup_path> - Rollback to specified backup")
        return
    
    command = sys.argv[1]
    
    if command == "create":
        backup_path = backup_system.create_backup()
        print(f"Backup created at: {backup_path}")
    
    elif command == "list":
        backups = backup_system.list_backups()
        if backups:
            print("Available backups:")
            for backup in backups:
                print(f"  {backup['timestamp']}: {backup['file_count']} files - {backup['path']}")
        else:
            print("No backups found")
    
    elif command == "rollback":
        if len(sys.argv) < 3:
            print("Usage: python backup_system.py rollback <backup_path>")
            return
        
        backup_path = sys.argv[2]
        success = backup_system.rollback(backup_path)
        if success:
            print("Rollback completed successfully")
        else:
            print("Rollback failed")
    
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()