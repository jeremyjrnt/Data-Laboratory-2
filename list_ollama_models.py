#!/usr/bin/env python3
"""
Script to list available LLM models on Ollama server.
"""

import subprocess
import json
import sys

def list_ollama_models():
    """List all available models on the Ollama server."""
    try:
        print("🔍 Checking Ollama models...")
        
        # Call ollama list command
        cmd = ["ollama", "list"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            print(f"❌ Error running 'ollama list': {result.stderr}")
            return
        
        output = result.stdout.strip()
        if not output:
            print("📭 No models found or Ollama not running")
            return
        
        print("✅ Available Ollama models:")
        print("=" * 60)
        print(output)
        print("=" * 60)
        
        # Parse and extract model names only
        lines = output.split('\n')[1:]  # Skip header
        model_names = []
        
        for line in lines:
            if line.strip():
                # Extract model name (first column)
                parts = line.split()
                if parts:
                    model_name = parts[0]
                    model_names.append(model_name)
        
        if model_names:
            print(f"\n🎯 Model names for use in --llm parameter:")
            for name in model_names:
                print(f"  - {name}")
        
    except subprocess.TimeoutExpired:
        print("❌ Timeout: Ollama server might not be running")
    except FileNotFoundError:
        print("❌ Ollama command not found. Is Ollama installed?")
    except Exception as e:
        print(f"❌ Error: {e}")

def check_ollama_server():
    """Check if Ollama server is running by making an API call."""
    try:
        print("\n🔧 Checking Ollama server status...")
        
        cmd = [
            "curl", "-s", "-X", "GET", 
            "http://localhost:11434/api/tags"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            print(f"❌ Ollama server not accessible: {result.stderr}")
            return
        
        try:
            response = json.loads(result.stdout)
            models = response.get("models", [])
            
            print("✅ Ollama server is running!")
            print(f"📊 Found {len(models)} models via API:")
            
            for model in models:
                name = model.get("name", "Unknown")
                size = model.get("size", 0)
                modified = model.get("modified_at", "Unknown")
                
                # Convert size to human readable
                size_mb = size / (1024 * 1024) if size else 0
                size_str = f"{size_mb:.1f} MB" if size_mb < 1024 else f"{size_mb/1024:.1f} GB"
                
                print(f"  🤖 {name} ({size_str})")
                
        except json.JSONDecodeError:
            print("⚠️ Could not parse server response")
            
    except subprocess.TimeoutExpired:
        print("❌ Timeout: Ollama server not responding")
    except FileNotFoundError:
        print("❌ curl command not found")
    except Exception as e:
        print(f"❌ Error checking server: {e}")

def main():
    """Main function."""
    print("🦙 Ollama Model Checker")
    print("=" * 40)
    
    # Method 1: Using ollama list command
    list_ollama_models()
    
    # Method 2: Using API call
    check_ollama_server()
    
    print("\n💡 Usage tips:")
    print("  - Use model names like: gemma3:4b, qwen3:8b, llama3.1:7b")
    print("  - Start Ollama with: ollama serve")
    print("  - Pull new models with: ollama pull <model_name>")

if __name__ == "__main__":
    main()