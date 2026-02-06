"""
Test script to validate UC extraction improvements
Run this after processing a PDF to verify all fixes are working
"""

def validate_extraction(structured_data):
    """
    Validate that all required fields and sections are properly extracted
    """
    issues = []
    warnings = []
    
    # Required keys that must be present
    REQUIRED_KEYS = [
        "Deposit Work ID",
        "Name of Work",
        "User Department",
        "Administrative Approval No",
        "Administrative Approval Date",
        "AA Amount in Rupees",
        "Sanctioned Amount",
        "Name of Executing Agency",
        "Tender Amount",
        "Work Order No",
        "Work Order Date",
        "Date of Start",
        "Stipulated Date of Completion",
        "Amount of Deposit"
    ]
    
    # Check 1: Heading
    if not structured_data.get("heading"):
        issues.append("❌ Missing heading")
    else:
        print("✅ Heading found:", structured_data["heading"])
    
    # Check 2: Two-column data (required keys)
    two_col_data = structured_data.get("two_column_data", [])
    found_keys = {item["label"]: item["value"] for item in two_col_data}
    
    print(f"\n✅ Found {len(found_keys)} keys in two-column data")
    
    for key in REQUIRED_KEYS:
        if key not in found_keys:
            issues.append(f"❌ Missing required key: {key}")
        elif not found_keys[key]:
            warnings.append(f"⚠️  Empty value for key: {key}")
        else:
            print(f"  ✓ {key}: {found_keys[key][:50]}...")
    
    # Check 3: Statements
    statements = structured_data.get("sentences", [])
    if not statements:
        issues.append("❌ No statements found")
    else:
        print(f"\n✅ Found {len(statements)} statements")
        for stmt in statements[:3]:  # Show first 3
            print(f"  ✓ {stmt[:60]}...")
    
    # Check 4: Receipt details
    receipt_details = structured_data.get("receipt_details", [])
    if not receipt_details:
        warnings.append("⚠️  No receipt details found")
    else:
        print(f"\n✅ Found {len(receipt_details)} receipt rows")
        if receipt_details:
            print(f"  Columns: {list(receipt_details[0].keys())}")
    
    # Check 5: Expenditure details
    expenditure_details = structured_data.get("expenditure_details", [])
    if not expenditure_details:
        warnings.append("⚠️  No expenditure details found")
    else:
        print(f"\n✅ Found {len(expenditure_details)} expenditure rows")
        if expenditure_details:
            columns = list(expenditure_details[0].keys())
            print(f"  Columns: {columns}")
            
            # Check for Remarks column
            has_remarks = any('remark' in str(col).lower() for col in columns)
            if has_remarks:
                print("  ✓ Remarks column present")
                # Check if remarks have content
                remarks_with_content = sum(1 for row in expenditure_details 
                                          if any(row.get(col) for col in columns 
                                                if 'remark' in str(col).lower()))
                print(f"  ✓ {remarks_with_content} rows with remarks content")
            else:
                warnings.append("⚠️  No Remarks column found in expenditure details")
    
    # Print summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    if not issues and not warnings:
        print("✅ ALL CHECKS PASSED - Extraction is complete and robust!")
    else:
        if issues:
            print(f"\n❌ CRITICAL ISSUES ({len(issues)}):")
            for issue in issues:
                print(f"  {issue}")
        
        if warnings:
            print(f"\n⚠️  WARNINGS ({len(warnings)}):")
            for warning in warnings:
                print(f"  {warning}")
    
    return len(issues) == 0


if __name__ == "__main__":
    print("This script is meant to be imported and used with extracted data")
    print("\nUsage:")
    print("  from test_extraction import validate_extraction")
    print("  validate_extraction(structured_data)")
