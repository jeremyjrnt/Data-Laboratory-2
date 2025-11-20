"""
FAISS Vector Database Inspector
Interactive tool to explore and manage vector databases
"""

import os
import json
import numpy as np
import faiss
from pathlib import Path
import random
from typing import Dict, List, Tuple, Optional
from config.config import Config
from config.config import Config

class VectorDBInspector:
    def __init__(self, vectordb_path: str = None):
        """
        Initialize the Vector DB Inspector
        
        Args:
            vectordb_path: Path to the VectorDBs directory
        """
        if vectordb_path is None:
            # Default to the project's VectorDBs directory from Config
            vectordb_path = Config.VECTORDB_DIR
        
        self.vectordb_path = Path(vectordb_path).resolve()
        self.current_db = None
        self.current_index = None
        self.current_metadata = None
        self.db_name = None
        
    def list_available_databases(self) -> List[str]:
        """List all available vector databases"""
        if not self.vectordb_path.exists():
            return []
        
        databases = []
        for file in self.vectordb_path.glob("*.index"):
            db_name = file.stem
            metadata_file = self.vectordb_path / f"{db_name}_metadata.json"
            if metadata_file.exists():
                databases.append(db_name)
        
        return databases
    
    def select_database(self) -> Optional[str]:
        """Interactive database selection"""
        databases = self.list_available_databases()
        
        if not databases:
            print("❌ No vector databases found in:", self.vectordb_path)
            return None
        
        print("\n🗂️  Available vector databases:")
        print("-" * 50)
        for i, db in enumerate(databases, 1):
            # Get database info
            index_file = self.vectordb_path / f"{db}.index"
            metadata_file = self.vectordb_path / f"{db}_metadata.json"
            
            size_mb = index_file.stat().st_size / (1024 * 1024)
            
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata_dict = json.load(f)
                # Handle both old format (direct list) and new format (dict with metadata key)
                if isinstance(metadata_dict, dict) and "metadata" in metadata_dict:
                    num_entries = len(metadata_dict["metadata"])
                else:
                    num_entries = len(metadata_dict)
            
            print(f"{i}. {db}")
            print(f"   📊 {num_entries:,} entries | 💾 {size_mb:.1f} MB")
        
        print("-" * 50)
        
        while True:
            try:
                choice = input(f"\nSelect a database (1-{len(databases)}) or 'q' to quit: ").strip()
                
                if choice.lower() == 'q':
                    return None
                
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(databases):
                    return databases[choice_idx]
                else:
                    print(f"❌ Please enter a number between 1 and {len(databases)}")
            
            except ValueError:
                print("❌ Please enter a valid number or 'q' to quit")
    
    def load_database(self, db_name: str) -> bool:
        """Load the selected database"""
        try:
            index_file = self.vectordb_path / f"{db_name}.index"
            metadata_file = self.vectordb_path / f"{db_name}_metadata.json"
            
            # Load FAISS index
            print(f"📥 Loading FAISS index: {index_file.name}")
            self.current_index = faiss.read_index(str(index_file))
            
            # Load metadata
            print(f"📥 Loading metadata: {metadata_file.name}")
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata_dict = json.load(f)
                # Extract the metadata list from the dictionary
                self.current_metadata = metadata_dict.get("metadata", metadata_dict)
            
            self.db_name = db_name
            print(f"✅ Database '{db_name}' loaded successfully")
            print(f"   📊 {len(self.current_metadata):,} entries")
            print(f"   🎯 Vector dimension: {self.current_index.d}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading database: {e}")
            return False
    
    def display_random_entries(self, num_entries: int = 5):
        """Display random entries with metadata"""
        if not self.current_metadata:
            print("❌ No database loaded")
            return
        
        print(f"\n🎲 Displaying {num_entries} random entries:")
        print("=" * 80)
        
        # Select random entries
        total_entries = len(self.current_metadata)
        random_indices = random.sample(range(total_entries), min(num_entries, total_entries))
        
        for i, idx in enumerate(random_indices, 1):
            entry = self.current_metadata[idx]
            print(f"\n📋 Entry {i} (ID: {idx})")
            print("-" * 40)
            
            # Display metadata
            for key, value in entry.items():
                if isinstance(value, str) and len(value) > 100:
                    # Truncate long strings
                    value = value[:100] + "..."
                elif isinstance(value, list) and len(value) > 3:
                    # Show first 3 items of long lists
                    value = value[:3] + [f"... (+{len(value)-3} more)"]
                
                print(f"  {key}: {value}")
        
        print("=" * 80)
    
    def search_similar_entries(self, entry_id: int, k: int = 5):
        """Find similar entries to a given entry"""
        if not self.current_index or not self.current_metadata:
            print("❌ No database loaded")
            return
        
        if entry_id >= len(self.current_metadata):
            print(f"❌ Invalid entry ID. Maximum: {len(self.current_metadata) - 1}")
            return
        
        try:
            # Get the vector for the specified entry
            query_vector = self.current_index.reconstruct(entry_id).reshape(1, -1)
            
            # Search for similar entries
            distances, indices = self.current_index.search(query_vector, k + 1)  # +1 to exclude self
            
            print(f"\n🔍 Entries similar to entry {entry_id}:")
            print("=" * 80)
            
            # Display query entry
            query_entry = self.current_metadata[entry_id]
            print(f"\n📋 Reference entry (ID: {entry_id})")
            print("-" * 40)
            for key, value in query_entry.items():
                if isinstance(value, str) and len(value) > 100:
                    value = value[:100] + "..."
                print(f"  {key}: {value}")
            
            print(f"\n🎯 Similar entries:")
            print("-" * 40)
            
            for i, (dist, idx) in enumerate(zip(distances[0][1:], indices[0][1:])):  # Skip first (self)
                if idx == -1:
                    continue
                
                entry = self.current_metadata[idx]
                similarity = float(dist)  # Cosine similarity (higher = more similar)
                
                print(f"\n{i+1}. ID: {idx} | Similarity: {similarity:.4f}")
                for key, value in entry.items():
                    if isinstance(value, str) and len(value) > 80:
                        value = value[:80] + "..."
                    print(f"     {key}: {value}")
            
            print("=" * 80)
            
        except Exception as e:
            print(f"❌ Error during search: {e}")
    
    def get_database_stats(self):
        """Display detailed statistics about the current database"""
        if not self.current_index or not self.current_metadata:
            print("❌ No database loaded")
            return
        
        print(f"\n📊 Database '{self.db_name}' statistics:")
        print("=" * 60)
        print(f"📈 Total entries: {len(self.current_metadata):,}")
        print(f"🎯 Vector dimension: {self.current_index.d}")
        print(f"💾 Index size: {self.current_index.ntotal:,} vectors")
        
        # Analyze metadata fields
        if self.current_metadata:
            sample_entry = self.current_metadata[0]
            print(f"🏷️  Available metadata fields:")
            for key in sample_entry.keys():
                print(f"   • {key}")
        
        # File sizes
        index_file = self.vectordb_path / f"{self.db_name}.index"
        metadata_file = self.vectordb_path / f"{self.db_name}_metadata.json"
        
        index_size = index_file.stat().st_size / (1024 * 1024)
        metadata_size = metadata_file.stat().st_size / (1024 * 1024)
        
        print(f"💾 Index file size: {index_size:.1f} MB")
        print(f"💾 Metadata file size: {metadata_size:.1f} MB")
        print(f"💾 Total size: {index_size + metadata_size:.1f} MB")
        print("=" * 60)
    
    def delete_database(self) -> bool:
        """Delete the current database (with confirmation)"""
        if not self.db_name:
            print("❌ No database loaded")
            return False
        
        print(f"\n⚠️  WARNING: You are about to delete the database '{self.db_name}'")
        print("This action is IRREVERSIBLE!")
        
        confirmation = input(f"Type exactly '{self.db_name}' to confirm deletion: ").strip()
        
        if confirmation != self.db_name:
            print("❌ Deletion cancelled (incorrect confirmation)")
            return False
        
        try:
            # Delete files
            index_file = self.vectordb_path / f"{self.db_name}.index"
            metadata_file = self.vectordb_path / f"{self.db_name}_metadata.json"
            
            index_file.unlink()
            metadata_file.unlink()
            
            print(f"✅ Database '{self.db_name}' deleted successfully")
            
            # Reset current state
            self.current_db = None
            self.current_index = None
            self.current_metadata = None
            self.db_name = None
            
            return True
            
        except Exception as e:
            print(f"❌ Error during deletion: {e}")
            return False
    
    def interactive_menu(self):
        """Main interactive menu"""
        while True:
            if not self.db_name:
                print("\n" + "=" * 60)
                print("🔍 VECTOR DATABASE INSPECTOR")
                print("=" * 60)
                
                # Select database
                selected_db = self.select_database()
                if not selected_db:
                    print("👋 Goodbye!")
                    break
                
                if not self.load_database(selected_db):
                    continue
            
            # Main menu
            print(f"\n🗂️  Active database: {self.db_name}")
            print("-" * 50)
            print("1. 🎲 Display 5 random entries")
            print("2. 📊 View database statistics")
            print("3. 🔍 Search for similar entries")
            print("4. 🎯 Display a specific entry")
            print("5. 🗑️  Delete this database")
            print("6. 🔄 Change database")
            print("7. ❌ Quit")
            print("-" * 50)
            
            choice = input("Your choice (1-7): ").strip()
            
            if choice == "1":
                self.display_random_entries()
            
            elif choice == "2":
                self.get_database_stats()
            
            elif choice == "3":
                try:
                    entry_id = int(input(f"Reference entry ID (0-{len(self.current_metadata)-1}): "))
                    num_similar = int(input("Number of similar entries to display (default: 5): ") or "5")
                    self.search_similar_entries(entry_id, num_similar)
                except ValueError:
                    print("❌ Please enter valid numbers")
            
            elif choice == "4":
                try:
                    entry_id = int(input(f"Entry ID to display (0-{len(self.current_metadata)-1}): "))
                    if 0 <= entry_id < len(self.current_metadata):
                        entry = self.current_metadata[entry_id]
                        print(f"\n📋 Entry {entry_id}:")
                        print("-" * 40)
                        for key, value in entry.items():
                            print(f"  {key}: {value}")
                    else:
                        print("❌ Invalid entry ID")
                except ValueError:
                    print("❌ Please enter a valid number")
            
            elif choice == "5":
                if self.delete_database():
                    # Database deleted, go back to selection
                    continue
            
            elif choice == "6":
                # Reset to select new database
                self.current_db = None
                self.current_index = None
                self.current_metadata = None
                self.db_name = None
            
            elif choice == "7":
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice, please select 1-7")
            
            input("\nPress Enter to continue...")


def main():
    """Main function"""
    inspector = VectorDBInspector()
    inspector.interactive_menu()


if __name__ == "__main__":
    main()
