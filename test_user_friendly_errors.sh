#!/bin/bash
# Quick Test Script for User-Friendly Errors
# Run this to verify everything is working

echo "🧪 Testing User-Friendly Error Messages..."
echo ""
echo "=========================================="
echo "TEST 1: Invalid File Type"
echo "=========================================="
echo ""
echo "Creating a text file..."
echo "This is not an audio file" > test_error.txt

echo "Uploading text file (should fail with user-friendly error)..."
response=$(curl -s -X POST http://localhost:8000/api/v1/upload \
  -F "file=@test_error.txt")

echo ""
echo "Response:"
echo "$response" | python3 -m json.tool

echo ""
echo "=========================================="
echo "VERIFICATION CHECKLIST:"
echo "=========================================="

# Check if response has user-friendly fields
if echo "$response" | grep -q '"icon"'; then
    echo "✅ Has icon field"
else
    echo "❌ Missing icon field"
fi

if echo "$response" | grep -q '"title"'; then
    echo "✅ Has title field"
else
    echo "❌ Missing title field"
fi

if echo "$response" | grep -q '"helpful_tips"'; then
    echo "✅ Has helpful_tips field"
else
    echo "❌ Missing helpful_tips field"
fi

if echo "$response" | grep -q '"error_id"'; then
    echo "✅ Has error_id field"
else
    echo "❌ Missing error_id field"
fi

if echo "$response" | grep -q 'ValidationError'; then
    echo "❌ Still showing technical 'ValidationError' - user-unfriendly"
else
    echo "✅ Not showing technical error names - user-friendly!"
fi

echo ""
echo "=========================================="
echo "EXPECTED OUTPUT:"
echo "=========================================="
cat << 'EOF'
{
  "success": false,
  "error": {
    "icon": "⚠️",
    "title": "Invalid Input",
    "message": "We can't process .txt files...",
    "severity": "warning",
    "error_id": "ERR_400_XXXX",
    "can_retry": false,
    "is_temporary": false,
    "helpful_tips": [
      "Check that your file is a valid audio format",
      "Ensure file size is under 100MB",
      "Make sure the file isn't corrupted"
    ],
    "timestamp": "...",
    "request_id": "..."
  },
  "status_code": 400
}
EOF

echo ""
echo "=========================================="
echo "Cleanup..."
rm -f test_error.txt
echo "Done!"
