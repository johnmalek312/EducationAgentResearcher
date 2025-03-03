import subprocess
import re
import requests
import os
import shutil
import argparse
from packaging import version
from datetime import datetime

def get_latest_version(package_name):
    """Get the latest version of a package from PyPI"""
    try:
        response = requests.get(f"https://pypi.org/pypi/{package_name}/json")
        if response.status_code == 200:
            data = response.json()
            return data["info"]["version"]
        return "Unknown"
    except Exception as e:
        print(f"Error checking {package_name}: {str(e)}")
        return "Error"

def parse_requirements_file(file_path="requirements.txt"):
    """Parse requirements.txt file and return the content and packages info"""
    packages = []
    lines = []
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            clean_line = line.strip()
            
            # Skip empty lines and comments
            if not clean_line or clean_line.startswith('#'):
                continue
                
            # Extract package name and version if specified
            match = re.match(r'^([a-zA-Z0-9_\-\.]+)(?:[=<>!~]+([a-zA-Z0-9_\.\-]+))?.*$', clean_line)
            if match:
                package_name = match.group(1)
                current_version = match.group(2) if match.group(2) else None
                packages.append({
                    'name': package_name,
                    'version': current_version,
                    'line_number': i,
                    'line': clean_line
                })
    except Exception as e:
        print(f"Error reading requirements file: {str(e)}")
        
    return lines, packages

def create_backup(file_path="requirements.txt"):
    """Create a backup of the requirements file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{file_path}.{timestamp}.bak"
    shutil.copy2(file_path, backup_path)
    print(f"Created backup at {backup_path}")
    return backup_path

def update_requirements_file(file_path="requirements.txt", packages_to_update=None, dry_run=False):
    """Update the requirements.txt file with latest versions"""
    if not packages_to_update:
        return
    
    lines, _ = parse_requirements_file(file_path)
    
    # Only create backup if we're actually updating
    if not dry_run:
        backup_path = create_backup(file_path)
    
    # Update lines with new versions
    for package in packages_to_update:
        line_number = package['line_number']
        original_line = lines[line_number].rstrip()
        
        # Find where to insert the version
        package_name = package['name']
        if '==' in original_line:
            updated_line = re.sub(
                fr'{package_name}==[\d\.]+', 
                f"{package_name}=={package['latest_version']}", 
                original_line
            )
        else:
            # Add version if it wasn't specified
            updated_line = re.sub(
                package_name, 
                f"{package_name}=={package['latest_version']}", 
                original_line
            )
        
        lines[line_number] = updated_line + '\n'
        
        print(f"{'[DRY RUN] ' if dry_run else ''}Updating: {original_line} -> {updated_line}")
    
    # Write updated content back to the file
    if not dry_run:
        try:
            with open(file_path, 'w') as f:
                f.writelines(lines)
            print(f"Successfully updated {file_path}")
        except Exception as e:
            print(f"Error writing to {file_path}: {str(e)}")
            # Try to restore from backup
            shutil.copy2(backup_path, file_path)
            print(f"Restored original file from backup")

def main():
    parser = argparse.ArgumentParser(description='Check and update package versions in requirements.txt')
    parser.add_argument('--update', action='store_true', help='Update requirements.txt with latest versions')
    parser.add_argument('--auto-yes', '-y', action='store_true', help='Automatically confirm updates')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be updated without making changes')
    parser.add_argument('--file', default='requirements.txt', help='Path to requirements file')
    
    args = parser.parse_args()
    
    print("Checking package versions...")
    lines, packages = parse_requirements_file(args.file)
    
    print("\n{:<30} {:<15} {:<15} {:<10}".format("Package", "Current", "Latest", "Status"))
    print("-" * 75)
    
    packages_to_update = []
    
    for package in packages:
        package_name = package['name']
        current_version = package['version'] or "Not specified"
        latest_version = get_latest_version(package_name)
        
        status = "Up to date"
        needs_update = False
        try:
            if current_version != "Not specified" and latest_version != "Unknown" and latest_version != "Error":
                if version.parse(latest_version) > version.parse(current_version):
                    status = "Update available"
                    needs_update = True
            elif current_version == "Not specified" and latest_version != "Unknown" and latest_version != "Error":
                status = "Version not specified"
                needs_update = True
        except Exception:
            status = "Version comparison error"
            
        print("{:<30} {:<15} {:<15} {:<10}".format(
            package_name, current_version, latest_version, status
        ))
        
        if needs_update:
            package['latest_version'] = latest_version
            packages_to_update.append(package)
    
    if args.update or args.dry_run:
        if packages_to_update:
            print(f"\nFound {len(packages_to_update)} packages to update")
            
            if args.dry_run:
                update_requirements_file(args.file, packages_to_update, dry_run=True)
            elif args.auto_yes or input("\nDo you want to update these packages? [y/N]: ").lower() == 'y':
                update_requirements_file(args.file, packages_to_update)
            else:
                print("Update cancelled")
        else:
            print("\nAll packages are up to date!")

if __name__ == "__main__":
    main()