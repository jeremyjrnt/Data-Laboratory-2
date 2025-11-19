#!/usr/bin/env python3
"""
Test script to check Ollama remote connection and LLM calls
"""

import json
import subprocess
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def test_ollama_remote():
    """Test connection to remote Ollama server."""
    
    print("🔍 Testing Ollama Remote Connection")
    print("=" * 50)
    
    # Configuration - Get from environment variables
    ollama_url = os.getenv('OLLAMA_REMOTE_A5000', 'http://100.64.0.7:11434')
    model_name = os.getenv('OLLAMA_MODEL', 'gpt-oss:20b')
    test_prompt = "Hello, how are you today? Please respond in one sentence."
    
    print(f"🌐 Server URL: {ollama_url}")
    print(f"🤖 Model: {model_name}")
    print(f"📝 Test prompt: {test_prompt}")
    print("-" * 50)
    
    # Test 1: Check if server is reachable
    print("📡 Test 1: Server connectivity...")
    try:
        # Simple curl to check if server responds
        cmd_ping = [
            "curl", "-s", "-f", 
            f"{ollama_url}/api/tags",
            "--connect-timeout", "10"
        ]
        
        result = subprocess.run(cmd_ping, capture_output=True, text=True, timeout=15)
        
        if result.returncode == 0:
            print("✅ Server is reachable")
            try:
                response_data = json.loads(result.stdout)
                models = response_data.get('models', [])
                print(f"📋 Available models: {len(models)} found")
                for model in models[:10]:  # Show first 5 models
                    print(f"   - {model.get('name', 'Unknown')}")
                if len(models) > 5:
                    print(f"   ... and {len(models) - 5} more")
            except json.JSONDecodeError:
                print("⚠️  Server responded but with invalid JSON")
        else:
            print(f"❌ Server not reachable - Status code: {result.returncode}")
            print(f"   Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ Server connection timeout")
        return False
    except Exception as e:
        print(f"❌ Server connection error: {e}")
        return False
    
    # Test 2: Test LLM generation
    print(f"\n🤖 Test 2: LLM generation with {model_name}...")
    
    try:
        payload = {
            "model": model_name,
            "prompt": test_prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9
            }
        }
        
        cmd_generate = [
            "curl", "-s", "-X", "POST",
            f"{ollama_url}/api/generate",
            "-H", "Content-Type: application/json",
            "-d", json.dumps(payload),
            "--connect-timeout", "10"
        ]
        
        print(f"🔄 Sending request...")
        start_time = time.time()
        
        result = subprocess.run(cmd_generate, capture_output=True, text=True, timeout=120)
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Request successful ({elapsed_time:.2f}s)")
            
            try:
                response_data = json.loads(result.stdout)
                llm_response = response_data.get('response', '').strip()
                
                if llm_response:
                    print(f"🎯 LLM Response:")
                    print(f"   \"{llm_response}\"")
                    
                    # Additional response info
                    if 'done' in response_data:
                        print(f"📊 Status: {'Complete' if response_data['done'] else 'Incomplete'}")
                    
                    if 'total_duration' in response_data:
                        duration_ms = response_data['total_duration'] / 1_000_000  # Convert to ms
                        print(f"⏱️  Server duration: {duration_ms:.0f}ms")
                    
                    print("✅ LLM call successful!")
                    return True
                else:
                    print("❌ Empty response from LLM")
                    print(f"Raw response: {result.stdout}")
                    return False
                    
            except json.JSONDecodeError as e:
                print(f"❌ Invalid JSON response: {e}")
                print(f"Raw response: {result.stdout[:500]}...")
                return False
                
        else:
            print(f"❌ Request failed - Status code: {result.returncode}")
            print(f"   Error: {result.stderr}")
            print(f"   Response: {result.stdout}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ LLM request timeout (120s)")
        return False
    except Exception as e:
        print(f"❌ LLM request error: {e}")
        return False

def test_multiple_calls():
    """Test multiple LLM calls to check consistency."""
    
    print(f"\n🔄 Test 3: Multiple LLM calls consistency...")
    
    ollama_url = os.getenv('OLLAMA_REMOTE_A5000', 'http://100.64.0.7:11434')
    model_name = os.getenv('OLLAMA_MODEL', 'gemma3:4b')
    
    test_prompts = [
        "What is 2+2?",
        "Name one color.",
        "Say hello in French.",
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n📝 Call {i}/3: \"{prompt}\"")
        
        try:
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9
                }
            }
            
            cmd = [
                "curl", "-s", "-X", "POST",
                f"{ollama_url}/api/generate",
                "-H", "Content-Type: application/json",
                "-d", json.dumps(payload),
                "--connect-timeout", "5"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                try:
                    response_data = json.loads(result.stdout)
                    llm_response = response_data.get('response', '').strip()
                    print(f"   ✅ \"{llm_response}\"")
                except json.JSONDecodeError:
                    print(f"   ❌ JSON decode error")
            else:
                print(f"   ❌ Request failed")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        time.sleep(1)  # Small delay between requests

if __name__ == "__main__":
    print("🧪 OLLAMA REMOTE CONNECTION TEST")
    print("=" * 60)
    
    # Run basic test
    success = test_ollama_remote()
    
    if success:
        # Run multiple calls test
        test_multiple_calls()
        
        print(f"\n🎉 All tests completed!")
        print("=" * 60)
    else:
        print(f"\n❌ Basic test failed - skipping additional tests")
        print("=" * 60)
