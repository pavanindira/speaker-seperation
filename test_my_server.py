#!/usr/bin/env python3
"""
Automated Test Script for speak.rigverse.in
Run this from YOUR computer (since you can access the server)
"""

import requests
import json
import sys
from pathlib import Path
import time

API_BASE = "https://speak.rigverse.in"

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

def print_test(name, status, details=""):
    icon = "✅" if status else "❌"
    color = Colors.GREEN if status else Colors.RED
    print(f"{icon} {color}{name}{Colors.RESET}")
    if details:
        print(f"   {details}")

def print_section(title):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}\n")

# ===========================================================================
# TEST SUITE
# ===========================================================================

def test_root_endpoint():
    """Test root endpoint"""
    print_section("TEST 1: Root Endpoint")
    try:
        response = requests.get(f"{API_BASE}/", timeout=10)
        print_test("Root Endpoint", response.status_code in [200, 307], 
                   f"Status: {response.status_code}")
        return response.status_code in [200, 307]
    except Exception as e:
        print_test("Root Endpoint", False, f"Error: {str(e)}")
        return False

def test_health():
    """Test health endpoint"""
    print_section("TEST 2: Health Check")
    try:
        response = requests.get(f"{API_BASE}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print_test("Health Endpoint", True, 
                      f"Response: {json.dumps(data, indent=2)}")
            return True
        else:
            print_test("Health Endpoint", False, 
                      f"Status: {response.status_code}")
            return False
    except Exception as e:
        print_test("Health Endpoint", False, f"Error: {str(e)}")
        return False

def test_frontend():
    """Test frontend UI"""
    print_section("TEST 3: Frontend UI")
    try:
        response = requests.get(f"{API_BASE}/ui", timeout=10)
        if response.status_code == 200:
            has_content = "Speaker Separation" in response.text
            print_test("Frontend UI Loads", has_content,
                      f"Found 'Speaker Separation': {has_content}")
            
            # Check for key elements
            has_upload = "upload" in response.text.lower()
            has_settings = "settings" in response.text.lower()
            
            print_test("Upload Area Present", has_upload)
            print_test("Settings Present", has_settings)
            
            return has_content
        else:
            print_test("Frontend UI", False, 
                      f"Status: {response.status_code}")
            return False
    except Exception as e:
        print_test("Frontend UI", False, f"Error: {str(e)}")
        return False

def test_docs():
    """Test API documentation"""
    print_section("TEST 4: API Documentation")
    try:
        response = requests.get(f"{API_BASE}/docs", timeout=10)
        print_test("API Docs", response.status_code == 200,
                  f"Status: {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print_test("API Docs", False, f"Error: {str(e)}")
        return False

def test_user_friendly_errors():
    """Test user-friendly error messages"""
    print_section("TEST 5: User-Friendly Error Messages")
    
    # Create test file
    test_file = Path("test_auto.txt")
    test_file.write_text("This is not an audio file for automated testing")
    
    try:
        with open(test_file, 'rb') as f:
            files = {'file': ('test.txt', f, 'text/plain')}
            response = requests.post(f"{API_BASE}/api/v1/upload", 
                                   files=files, timeout=10)
        
        test_file.unlink()
        
        if response.status_code == 400:
            data = response.json()
            
            # Check new error format
            has_error_obj = 'error' in data and isinstance(data['error'], dict)
            has_icon = has_error_obj and 'icon' in data['error']
            has_title = has_error_obj and 'title' in data['error']
            has_tips = has_error_obj and 'helpful_tips' in data['error']
            has_error_id = has_error_obj and 'error_id' in data['error']
            
            print_test("Error Object Structure", has_error_obj)
            print_test("Has Icon", has_icon, 
                      f"Icon: {data.get('error', {}).get('icon', 'N/A')}")
            print_test("Has Title", has_title,
                      f"Title: {data.get('error', {}).get('title', 'N/A')}")
            print_test("Has Helpful Tips", has_tips,
                      f"Tips: {len(data.get('error', {}).get('helpful_tips', []))} tips")
            print_test("Has Error ID", has_error_id,
                      f"ID: {data.get('error', {}).get('error_id', 'N/A')}")
            
            if not has_tips:
                print(f"\n{Colors.YELLOW}⚠️  Error format is old/incomplete:{Colors.RESET}")
                print(f"   Current: {json.dumps(data, indent=2)}")
            
            return has_error_obj and has_tips and has_error_id
        else:
            print_test("Error Handling", False,
                      f"Expected 400, got {response.status_code}")
            return False
            
    except Exception as e:
        print_test("Error Test", False, f"Error: {str(e)}")
        if test_file.exists():
            test_file.unlink()
        return False

def test_file_size_limit():
    """Test file size validation"""
    print_section("TEST 6: File Size Limit")
    
    # Create 101MB file
    print("   Creating 101MB test file...")
    test_file = Path("test_large_auto.mp3")
    
    try:
        with open(test_file, 'wb') as f:
            f.write(b'0' * (101 * 1024 * 1024))
        
        print("   Uploading large file...")
        with open(test_file, 'rb') as f:
            files = {'file': ('large.mp3', f, 'audio/mpeg')}
            response = requests.post(f"{API_BASE}/api/v1/upload",
                                   files=files, timeout=30)
        
        test_file.unlink()
        
        if response.status_code == 413:
            data = response.json()
            has_helpful_message = 'error' in data and 'helpful_tips' in data.get('error', {})
            
            print_test("File Size Limit Enforced", True,
                      "Large file rejected correctly")
            print_test("Has Helpful Tips", has_helpful_message,
                      "Tips about compression/splitting")
            
            if has_helpful_message:
                tips = data['error']['helpful_tips']
                print(f"\n   Helpful tips provided:")
                for tip in tips:
                    print(f"   → {tip}")
            
            return True
        else:
            print_test("File Size Limit", False,
                      f"Expected 413, got {response.status_code}")
            return False
            
    except Exception as e:
        print_test("File Size Test", False, f"Error: {str(e)}")
        if test_file.exists():
            test_file.unlink()
        return False

def test_cors():
    """Test CORS configuration"""
    print_section("TEST 7: CORS Headers")
    try:
        response = requests.options(f"{API_BASE}/api/v1/upload", timeout=10)
        headers = response.headers
        
        has_cors = 'access-control-allow-origin' in headers
        origin = headers.get('access-control-allow-origin', 'Not set')
        
        print_test("CORS Configured", has_cors,
                  f"Allow-Origin: {origin}")
        return has_cors
    except Exception as e:
        print_test("CORS Test", False, f"Error: {str(e)}")
        return False

def test_ssl():
    """Test SSL certificate"""
    print_section("TEST 8: SSL Certificate")
    try:
        response = requests.get(f"{API_BASE}/", timeout=10, verify=True)
        print_test("SSL Certificate", True, "Valid HTTPS certificate")
        return True
    except requests.exceptions.SSLError:
        print_test("SSL Certificate", False, "Invalid or expired certificate")
        return False
    except Exception as e:
        print_test("SSL Test", False, f"Error: {str(e)}")
        return False

def test_response_times():
    """Test response times"""
    print_section("TEST 9: Response Times")
    
    endpoints = [
        ("/health", "Health Check"),
        ("/ui", "Frontend UI"),
        ("/docs", "API Docs")
    ]
    
    all_fast = True
    for endpoint, name in endpoints:
        try:
            start = time.time()
            response = requests.get(f"{API_BASE}{endpoint}", timeout=10)
            duration = time.time() - start
            
            is_fast = duration < 2.0  # Should be under 2 seconds
            print_test(f"{name} Response Time", is_fast,
                      f"{duration:.2f}s {'✅' if is_fast else '⚠️ Slow'}")
            all_fast = all_fast and is_fast
        except Exception as e:
            print_test(f"{name} Response Time", False, f"Error: {str(e)}")
            all_fast = False
    
    return all_fast

def check_frontend_features():
    """Check frontend has expected features"""
    print_section("TEST 10: Frontend Feature Check")
    
    try:
        response = requests.get(f"{API_BASE}/ui", timeout=10)
        html = response.text.lower()
        
        features = {
            "Upload Area": "upload" in html or "drop" in html,
            "Settings": "settings" in html or "speakers" in html,
            "Help Section": "help" in html or "tips" in html,
            "Error Display": "error" in html,
            "Progress Indicator": "progress" in html or "loading" in html
        }
        
        for feature, present in features.items():
            print_test(feature, present)
        
        return all(features.values())
        
    except Exception as e:
        print_test("Frontend Features", False, f"Error: {str(e)}")
        return False

# ===========================================================================
# MAIN TEST RUNNER
# ===========================================================================

def run_all_tests():
    """Run all tests and generate report"""
    
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}")
    print(f"🧪 Speaker Separation API - Automated Test Suite")
    print(f"{'='*70}{Colors.RESET}\n")
    print(f"Testing: {Colors.YELLOW}{API_BASE}{Colors.RESET}\n")
    
    results = []
    
    # Run tests
    results.append(("Root Endpoint", test_root_endpoint()))
    results.append(("Health Check", test_health()))
    results.append(("Frontend UI", test_frontend()))
    results.append(("API Documentation", test_docs()))
    results.append(("User-Friendly Errors", test_user_friendly_errors()))
    results.append(("File Size Limit", test_file_size_limit()))
    results.append(("CORS Headers", test_cors()))
    results.append(("SSL Certificate", test_ssl()))
    results.append(("Response Times", test_response_times()))
    results.append(("Frontend Features", check_frontend_features()))
    
    # Print summary
    print_summary(results)

def print_summary(results):
    """Print test summary with recommendations"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}")
    print("📊 TEST SUMMARY")
    print(f"{'='*70}{Colors.RESET}\n")
    
    passed = sum(1 for _, result in results if result is True)
    failed = sum(1 for _, result in results if result is False)
    total = len(results)
    
    print(f"✅ Passed:  {Colors.GREEN}{passed}{Colors.RESET}/{total}")
    print(f"❌ Failed:  {Colors.RED}{failed}{Colors.RESET}/{total}")
    
    success_rate = (passed / total * 100) if total > 0 else 0
    color = Colors.GREEN if success_rate >= 80 else Colors.YELLOW if success_rate >= 60 else Colors.RED
    print(f"\n📈 Success Rate: {color}{success_rate:.1f}%{Colors.RESET}")
    
    # Show failed tests
    if failed > 0:
        print(f"\n{Colors.YELLOW}🔧 Failed Tests:{Colors.RESET}")
        for name, result in results:
            if result is False:
                print(f"  ❌ {name}")
        
        print(f"\n{Colors.BLUE}💡 Recommendations:{Colors.RESET}")
        
        # Specific recommendations based on failures
        failures = [name for name, result in results if not result]
        
        if "User-Friendly Errors" in failures:
            print(f"  • Update error_handler.py with UserFriendlyErrorFormatter")
            print(f"  • Add exception handlers to final_speaker_api.py")
        
        if "Frontend UI" in failures:
            print(f"  • Check frontend file exists (final_speaker_frontend_improved.html)")
            print(f"  • Add @app.get('/ui') route")
        
        if "CORS Headers" in failures:
            print(f"  • Add CORS middleware: app.add_middleware(CORSMiddleware, ...)")
        
        if "Response Times" in failures:
            print(f"  • Check server resources (CPU, memory)")
            print(f"  • Consider caching or optimization")
    
    # Overall assessment
    print(f"\n{Colors.BOLD}Overall Assessment:{Colors.RESET}")
    if success_rate == 100:
        print(f"{Colors.GREEN}🎉 Perfect! Your API is production-ready!{Colors.RESET}")
        print(f"\nNext steps:")
        print(f"  • Monitor logs: tail -f logs/app.log")
        print(f"  • Set up monitoring/alerting")
        print(f"  • Test with real users")
        print(f"  • Consider adding features from COMPREHENSIVE_IMPROVEMENTS.md")
    elif success_rate >= 80:
        print(f"{Colors.GREEN}✅ Great! Almost production-ready.{Colors.RESET}")
        print(f"   Fix the {failed} failing test(s) and you're good to go!")
    elif success_rate >= 60:
        print(f"{Colors.YELLOW}⚠️  Good foundation, but needs work.{Colors.RESET}")
        print(f"   Address the failing tests before going live.")
    else:
        print(f"{Colors.RED}❌ Significant issues found.{Colors.RESET}")
        print(f"   Critical fixes needed before production use.")
    
    print(f"\n{Colors.BLUE}📚 Resources:{Colors.RESET}")
    print(f"  • Manual Testing Guide: MANUAL_TESTING_GUIDE.md")
    print(f"  • Troubleshooting: TROUBLESHOOTING_GUIDE.md")
    print(f"  • Improvements: COMPREHENSIVE_IMPROVEMENTS.md")

if __name__ == "__main__":
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Tests interrupted by user{Colors.RESET}")
    except Exception as e:
        print(f"\n{Colors.RED}Unexpected error: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()
