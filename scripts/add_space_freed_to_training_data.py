#!/usr/bin/env python3
"""
Script to add total space freed metrics to claude_v2_training JSON files.

This script:
1. Reads LAST_RUN_LOG.json to get total space_freed_kb for the episode
2. Updates all episode step files to include this metric
3. Can be run after episodes complete to enrich training data
"""

import json
import glob
import os
from pathlib import Path
from typing import Dict, List
import argparse
from datetime import datetime

def get_space_freed_from_log(log_file_path: str) -> float:
    """Extract space_freed_kb from LAST_RUN_LOG.json."""
    try:
        with open(log_file_path, 'r') as f:
            log_data = json.load(f)
        return log_data.get('space_freed_kb', 0.0)
    except Exception as e:
        print(f"Error reading {log_file_path}: {e}")
        return 0.0

def get_episode_id_from_filename(filename: str) -> str:
    """Extract episode ID from step filename."""
    # Example: nogit-285761-000000_20260228-5e3uc2_episode_v2_step_1_2026-02-28-21-39-04.json
    parts = filename.split('_')
    for i, part in enumerate(parts):
        if part.startswith('202') and len(part) > 10:  # Find the date-id part
            return part
    return ""

def update_training_files(training_dir: str, dry_run: bool = False) -> Dict[str, int]:
    """Update all training JSON files with space_freed_kb."""
    training_path = Path(training_dir)
    
    # Find LAST_RUN_LOG.json
    log_file = training_path / "LAST_RUN_LOG.json"
    if not log_file.exists():
        print(f"No LAST_RUN_LOG.json found in {training_dir}")
        return {"updated": 0, "skipped": 0, "errors": 0}
    
    # Get total space freed
    total_space_freed = get_space_freed_from_log(str(log_file))
    print(f"Total space freed from log: {total_space_freed} KB")
    
    # Find all episode step files
    step_files = list(training_path.glob("*episode_v2_step_*.json"))
    print(f"Found {len(step_files)} step files")
    
    stats = {"updated": 0, "skipped": 0, "errors": 0}
    
    for step_file in step_files:
        try:
            # Read existing data
            with open(step_file, 'r') as f:
                data = json.load(f)
            
            # Check if already has space_freed_kb
            if 'space_freed_kb' in data:
                print(f"  {step_file.name} - already has space_freed_kb, skipping")
                stats["skipped"] += 1
                continue
            
            # Add space freed metric
            data['space_freed_kb'] = total_space_freed
            data['space_freed_updated'] = datetime.now().isoformat()
            
            if not dry_run:
                # Write back to file
                with open(step_file, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"  ✅ {step_file.name} - added space_freed_kb: {total_space_freed} KB")
            else:
                print(f"  [DRY RUN] Would add space_freed_kb: {total_space_freed} KB to {step_file.name}")
            
            stats["updated"] += 1
            
        except Exception as e:
            print(f"  ❌ Error updating {step_file.name}: {e}")
            stats["errors"] += 1
    
    return stats

def update_all_episodes(training_dir: str, dry_run: bool = False) -> Dict[str, int]:
    """Update training files for all episodes in the directory."""
    training_path = Path(training_dir)
    
    # Group files by episode
    episode_groups = {}
    
    # Find all step files
    step_files = list(training_path.glob("*episode_v2_step_*.json"))
    
    for step_file in step_files:
        # Extract episode identifier from filename
        filename = step_file.name
        # Look for pattern like "20260228-5e3uc2"
        parts = filename.split('_')
        episode_id = None
        
        for part in parts:
            if '-' in part and len(part) > 8:  # Date-hash pattern
                episode_id = part
                break
        
        if episode_id:
            if episode_id not in episode_groups:
                episode_groups[episode_id] = []
            episode_groups[episode_id].append(step_file)
    
    print(f"Found {len(episode_groups)} episode groups")
    
    total_stats = {"updated": 0, "skipped": 0, "errors": 0}
    
    for episode_id, files in episode_groups.items():
        print(f"\nProcessing episode {episode_id} ({len(files)} files)")
        
        # Look for corresponding LAST_RUN_LOG.json or episode-specific log
        log_candidates = [
            training_path / "LAST_RUN_LOG.json",
            training_path / f"LAST_RUN_LOG_{episode_id}.json",
        ]
        
        space_freed = 0.0
        log_found = False
        
        for log_candidate in log_candidates:
            if log_candidate.exists():
                space_freed = get_space_freed_from_log(str(log_candidate))
                log_found = True
                print(f"  Using log: {log_candidate.name} (space_freed: {space_freed} KB)")
                break
        
        if not log_found:
            print(f"  ⚠️  No log file found for episode {episode_id}, using 0.0 KB")
        
        # Update all files for this episode
        for step_file in files:
            try:
                with open(step_file, 'r') as f:
                    data = json.load(f)
                
                if 'space_freed_kb' in data:
                    print(f"    {step_file.name} - already has space_freed_kb, skipping")
                    total_stats["skipped"] += 1
                    continue
                
                data['space_freed_kb'] = space_freed
                data['space_freed_updated'] = datetime.now().isoformat()
                
                if not dry_run:
                    with open(step_file, 'w') as f:
                        json.dump(data, f, indent=2)
                    print(f"    ✅ {step_file.name} - added space_freed_kb: {space_freed} KB")
                else:
                    print(f"    [DRY RUN] Would add space_freed_kb: {space_freed} KB to {step_file.name}")
                
                total_stats["updated"] += 1
                
            except Exception as e:
                print(f"    ❌ Error updating {step_file.name}: {e}")
                total_stats["errors"] += 1
    
    return total_stats

def main():
    parser = argparse.ArgumentParser(description="Add space_freed_kb to training JSON files")
    parser.add_argument(
        "training_dir", 
        nargs="?",
        default="../data/claude_v2_training",
        help="Directory containing training JSON files (default: ../data/claude_v2_training)"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Preview changes without modifying files"
    )
    parser.add_argument(
        "--current-episode-only",
        action="store_true",
        help="Only update files for the current episode (uses LAST_RUN_LOG.json)"
    )
    
    args = parser.parse_args()
    
    training_dir = args.training_dir
    
    print(f"Processing training data directory: {training_dir}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 60)
    
    if args.current_episode_only:
        stats = update_training_files(training_dir, args.dry_run)
    else:
        stats = update_all_episodes(training_dir, args.dry_run)
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"  Files updated: {stats['updated']}")
    print(f"  Files skipped: {stats['skipped']}")
    print(f"  Errors: {stats['errors']}")
    
    if args.dry_run:
        print("\n[DRY RUN] No files were actually modified.")
        print("Run without --dry-run to apply changes.")

if __name__ == "__main__":
    main()
