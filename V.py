def decode_vin_structure(vin: str):
    """
    Decodes basic vehicle info using the standardized VIN structure.
    NOTE: VIN must be a 17-character code. This is NOT RTO data.
    """
    if len(vin) != 17:
        return "❌ Error: VIN must be exactly 17 characters long."

    vin = vin.upper()

    # --- 1. World Manufacturer Identifier (WMI) - VIN positions 1-3 ---
    wmi = vin[0:3]

    # Simple lookup for common Indian/global WMI codes
    wmi_lookup = {
        'MAJ': 'Mahindra & Mahindra',
        'MA1': 'Ashok Leyland',
        'MA7': 'Hindustan Motors',
        'MEG': 'Mercedes-Benz India',
        'MBH': 'Honda India',
        'MHR': 'Hyundai India',
        'MBJ': 'Toyota Kirloskar Motor',
        'MPA': 'Tata Motors',
        # Other common WMI examples (for reference)
        'JHN': 'Honda (Japan)',
        '3GZ': 'Chevrolet (Mexico)',
        'WBA': 'BMW (Germany)',
        '1FM': 'Ford (USA)',
    }
    manufacturer = wmi_lookup.get(wmi, "Manufacturer Unknown / Not in lookup")

    # --- 2. Model Year - VIN position 10 ---
    year_code = vin[9]
    # Standardised code for model year (repeats every 30 years)
    year_lookup = {
        'A': 2010, 'B': 2011, 'C': 2012, 'D': 2013, 'E': 2014, 'F': 2015,
        'G': 2016, 'H': 2017, 'J': 2018, 'K': 2019, 'L': 2020, 'M': 2021,
        'N': 2022, 'P': 2023, 'R': 2024, 'S': 2025, 'T': 2026, 'V': 2027,
        'W': 2028, 'X': 2029, 'Y': 2030,
        # Older years (e.g., A=1980, B=1981, etc., up to Y=2000)
    }
    model_year = year_lookup.get(year_code, "Year Code Unknown")

    print(f"**--- VIN Structural Decode for {vin} ---**")
    print(f"**Manufacturer (WMI):** {manufacturer}")
    print(f"**Model Year (Approx):** {model_year}")
    print(f"**WMI Code (Positions 1-3):** {wmi}")
    print(f"**Vehicle Descriptor Section (VDS) (4-9):** {vin[3:9]}")
    print(f"**Vehicle Indicator Section (VIS) (12-17):** {vin[11:17]}")
    print("---------------------------------------------")

# Example VIN (You must input a 17-character VIN)
# Note: A vehicle registration number is NOT a VIN.
decode_vin_structure("MA3JMT31SMA412001")