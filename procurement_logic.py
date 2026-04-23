import datetime
import re

import pandas as pd


CURRENT_YEAR = datetime.datetime.now().year
LOCAL_MARKET_FILE = "master_market_data.csv"
LOCAL_STOCK_FILE = "combined_inventory.csv"
MASTER_CATALOG_FILE = "master_car_prices.csv"


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def first_existing_column(df, candidates):
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def normalize_text(value):
    if pd.isna(value):
        return ""
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def tokenize_variant(value):
    text = normalize_text(value)
    if not text:
        return set()
    stop_words = {"variant", "model", "edition", "petrol", "diesel", "automatic", "manual"}
    return {token for token in text.split() if token not in stop_words}


def variant_similarity_score(left, right):
    left_tokens = tokenize_variant(left)
    right_tokens = tokenize_variant(right)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = left_tokens & right_tokens
    union = left_tokens | right_tokens
    return len(overlap) / len(union) if union else 0.0


def parse_owner_rank(value):
    text = normalize_text(value)
    if not text:
        return pd.NA
    if "first" in text or "1st" in text:
        return 1
    if "second" in text or "2nd" in text:
        return 2
    if "third" in text or "3rd" in text:
        return 3
    if "fourth" in text or "4th" in text:
        return 4
    match = re.search(r"(\d+)", text)
    return int(match.group(1)) if match else pd.NA


def parse_listing_days(value):
    text = normalize_text(value)
    if not text:
        return pd.NA
    if "today" in text:
        return 0
    if "yesterday" in text:
        return 1
    if "day" in text:
        match = re.search(r"(\d+)", text)
        if match:
            return int(match.group(1))
    if "-" in str(value):
        parsed = pd.to_datetime(str(value), format="%b-%d", errors="coerce")
        if pd.notna(parsed):
            parsed = parsed.replace(year=CURRENT_YEAR)
            return max((pd.Timestamp.today().normalize() - parsed).days, 0)
    return pd.NA


def normalize_inventory_schema(df):
    if df.empty:
        return df

    df = df.copy()
    rename_map = {}

    for canonical, candidates in {
        "Make/Brand": ["Make/Brand", "Make", "Brand"],
        "Model": ["Model"],
        "Variant": ["Variant", "Version", "Trim"],
        "Location": ["Location", "City", "State"],
        "State": ["State", "Region"],
        "Listing_URL": ["Listing_URL", "Detail_URL", "URL"],
        "Source": ["Source", "Dealer", "Marketplace"],
        "Price_Raw": ["Price_Raw", "PriceRaw", "Price_Value"],
        "Kilometer": ["Kilometer", "KM", "Mileage"],
        "Reg_Year": ["Reg_Year", "Year", "Registration_Year"],
        "Age": ["Age"],
        "Fuel_Type": ["Fuel_Type", "Fuel"],
        "Transmission": ["Transmission"],
        "Status": ["Status"],
        "Owner": ["Owner", "Overview_Owner"],
        "Dealer_Name": ["Dealer_Name"],
        "Listing_Date": ["Listing_Date"],
    }.items():
        existing = first_existing_column(df, candidates)
        if existing and existing != canonical:
            rename_map[existing] = canonical

    if rename_map:
        df = df.rename(columns=rename_map)

    for text_col in [
        "Make/Brand",
        "Model",
        "Variant",
        "Location",
        "State",
        "Fuel_Type",
        "Transmission",
        "Source",
        "Status",
        "Owner",
        "Dealer_Name",
    ]:
        if text_col not in df.columns:
            df[text_col] = "Unknown"
        df[text_col] = df[text_col].fillna("Unknown").astype(str).str.strip()

    if "Listing_URL" not in df.columns:
        df["Listing_URL"] = ""
    if "Listing_Date" not in df.columns:
        df["Listing_Date"] = pd.NA

    if "Price_Raw" in df.columns:
        df["Price_Raw"] = safe_numeric(df["Price_Raw"])
    else:
        df["Price_Raw"] = pd.Series(dtype="float64")

    if "Kilometer" in df.columns:
        df["Kilometer"] = safe_numeric(df["Kilometer"])
        df.loc[df["Kilometer"] <= 0, "Kilometer"] = pd.NA
    else:
        df["Kilometer"] = pd.Series(dtype="float64")

    if "Reg_Year" in df.columns:
        df["Reg_Year"] = safe_numeric(df["Reg_Year"])
        df.loc[(df["Reg_Year"] < 1990) | (df["Reg_Year"] > CURRENT_YEAR), "Reg_Year"] = pd.NA
    else:
        df["Reg_Year"] = pd.Series(dtype="float64")

    if "Age" in df.columns:
        df["Age"] = safe_numeric(df["Age"])
    else:
        df["Age"] = pd.NA
    df["Age"] = df["Age"].fillna(CURRENT_YEAR - df["Reg_Year"])
    df.loc[df["Age"] < 0, "Age"] = 0

    df["Owner_Rank"] = df["Owner"].apply(parse_owner_rank)
    df["Listing_Days"] = df["Listing_Date"].apply(parse_listing_days)
    df["Make_Key"] = df["Make/Brand"].apply(normalize_text)
    df["Model_Key"] = df["Model"].apply(normalize_text)
    df["Variant_Key"] = df["Variant"].apply(normalize_text)
    df["Fuel_Key"] = df["Fuel_Type"].apply(normalize_text)
    df["Transmission_Key"] = df["Transmission"].apply(normalize_text)
    df["Price_Lakhs"] = df["Price_Raw"] / 100000
    df = df.dropna(subset=["Make/Brand", "Model", "Price_Raw"], how="any")
    return df


def normalize_catalog_schema(df):
    if df.empty:
        return df

    df = df.copy()
    df.columns = df.columns.str.strip()

    for col in ["Make", "Model", "Variant", "Market_Status", "Fuel_Type", "Transmission"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).str.strip()

    if "Ex_Showroom_Price" not in df.columns:
        df["Ex_Showroom_Price"] = pd.Series(dtype="float64")
    df["Ex_Showroom_Price"] = safe_numeric(df["Ex_Showroom_Price"])
    df.loc[df["Ex_Showroom_Price"] <= 0, "Ex_Showroom_Price"] = pd.NA
    df["Make_Key"] = df["Make"].apply(normalize_text)
    df["Model_Key"] = df["Model"].apply(normalize_text)
    df["Variant_Key"] = df["Variant"].apply(normalize_text)
    df["Fuel_Key"] = df["Fuel_Type"].apply(normalize_text)
    df["Transmission_Key"] = df["Transmission"].apply(normalize_text)
    return df


def load_csv_dataset(path, normalizer):
    try:
        return normalizer(pd.read_csv(path)), ""
    except FileNotFoundError:
        return pd.DataFrame(), f"{path} not found."
    except Exception as exc:
        return pd.DataFrame(), f"Failed to load {path}: {exc}"


def get_catalog_price(active_catalog, brand, model, variant, fuel_type, transmission, manual_price):
    if manual_price > 0:
        return manual_price, "Manual Input", "Unknown"

    if active_catalog.empty:
        return 0, "", "Unknown"

    pool = active_catalog[
        (active_catalog["Make_Key"] == normalize_text(brand))
        & (active_catalog["Model_Key"] == normalize_text(model))
    ].copy()

    fuel_key = normalize_text(fuel_type) if fuel_type != "Any Fuel" else ""
    transmission_key = normalize_text(transmission) if transmission != "Any Transmission" else ""
    variant_key = normalize_text(variant) if variant != "Any Variant" else ""

    if fuel_key:
        exact_fuel = pool[pool["Fuel_Key"] == fuel_key]
        if not exact_fuel.empty:
            pool = exact_fuel
    if transmission_key:
        exact_transmission = pool[pool["Transmission_Key"] == transmission_key]
        if not exact_transmission.empty:
            pool = exact_transmission
    if variant_key:
        exact_variant = pool[pool["Variant_Key"] == variant_key]
        if not exact_variant.empty:
            valid = exact_variant["Ex_Showroom_Price"].dropna()
            if not valid.empty:
                market_status = exact_variant["Market_Status"].iloc[0] if "Market_Status" in exact_variant.columns else "Unknown"
                return float(valid.iloc[0]), "Exact Master Catalog", market_status

    valid = pool["Ex_Showroom_Price"].dropna()
    if valid.empty:
        return 0, "", "Unknown"

    status_mode = pool["Market_Status"].mode()
    market_status = status_mode.iloc[0] if not status_mode.empty else "Unknown"
    return float(valid.mean()), "Catalog Average", market_status


def build_comparable_pool(market_df, brand, model, variant, year, location, fuel, transmission, owner_count, current_km):
    if market_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    base_pool = market_df[(market_df['Make/Brand'] == brand) & (market_df['Model'] == model)].copy()
    if base_pool.empty:
        return base_pool, base_pool, base_pool

    comp_pool = base_pool.copy()
    comp_pool['comp_weight'] = 1.0
    comp_pool['pricing_scope'] = "Broad Model Match"

    if year is not None and year != "Any Year":
        y = int(year)
        comp_pool = comp_pool[(comp_pool['Reg_Year'] >= y - 1) & (comp_pool['Reg_Year'] <= y + 1)]
        comp_pool.loc[comp_pool['Reg_Year'] == y, 'comp_weight'] += 1.0
        comp_pool.loc[comp_pool['Reg_Year'] == y, 'pricing_scope'] = "Exact Year Match"
        
    if variant != "Any Variant":
        comp_pool.loc[comp_pool['Variant'] == variant, 'comp_weight'] += 2.0
        comp_pool.loc[comp_pool['Variant'] == variant, 'pricing_scope'] = "Variant Match"
        
    if location != "All India":
        comp_pool.loc[comp_pool['Location'] == location, 'comp_weight'] += 0.5
        
    if fuel != "Any Fuel":
        comp_pool = comp_pool[comp_pool['Fuel_Type'] == fuel]
        
    if transmission != "Any Transmission":
        comp_pool = comp_pool[comp_pool['Transmission'] == transmission]

    weighted_pool = comp_pool[comp_pool['comp_weight'] >= 2.0] if not comp_pool.empty else pd.DataFrame()
    return base_pool, weighted_pool, comp_pool

def compute_confidence_score(exact_comps, strong_comps, km_coverage, owner_coverage, source_count, is_synthetic):
    if is_synthetic:
        return 20, "Low"

    score = 15
    score += min(exact_comps, 8) * 7
    score += min(strong_comps, 12) * 2
    score += min(source_count, 5) * 4
    score += int(km_coverage * 12)
    score += int(owner_coverage * 8)
    score = max(0, min(score, 100))
    if score >= 75:
        return score, "High"
    if score >= 50:
        return score, "Medium"
    return score, "Low"


def compute_demand_score(base_pool, comparable_pool):
    if base_pool.empty:
        return 0, "Unknown", "No market depth available."

    supply_count = len(comparable_pool)
    source_count = comparable_pool["Source"].nunique() if "Source" in comparable_pool.columns else 1
    median_days = comparable_pool["Listing_Days"].dropna().median() if "Listing_Days" in comparable_pool.columns else pd.NA

    score = 50
    score += max(0, 4 - supply_count) * 7
    score += min(source_count, 4) * 3

    if pd.notna(median_days):
        if median_days <= 7:
            score += 15
        elif median_days <= 21:
            score += 5
        elif median_days > 45:
            score -= 12

    if supply_count >= 12:
        score -= 15
    elif supply_count >= 8:
        score -= 8

    score = max(0, min(score, 100))
    if score >= 70:
        return score, "High Demand", "Market supply is tight or moving quickly."
    if score >= 45:
        return score, "Balanced", "The car has tradable demand, but pricing discipline still matters."
    return score, "Slow Demand", "Supply looks heavy or listings appear to move slowly."


def compute_internal_stock_signal(stock_df, brand, model, variant, fuel_type, transmission):
    if stock_df.empty:
        return 0, 0, "Internal stock file not available."

    base_stock = stock_df[
        (stock_df["Make_Key"] == normalize_text(brand))
        & (stock_df["Model_Key"] == normalize_text(model))
    ].copy()
    exact_stock = base_stock.copy()
    if variant != "Any Variant":
        exact_stock = exact_stock[exact_stock["Variant_Key"] == normalize_text(variant)]
    if fuel_type != "Any Fuel":
        exact_stock = exact_stock[exact_stock["Fuel_Key"] == normalize_text(fuel_type)]
    if transmission != "Any Transmission":
        exact_stock = exact_stock[exact_stock["Transmission_Key"] == normalize_text(transmission)]

    if len(exact_stock) > 0:
        note = f"You already have {len(exact_stock)} near-identical car(s) in internal stock."
    elif len(base_stock) > 0:
        note = f"You already have {len(base_stock)} car(s) of this model in internal stock."
    else:
        note = "This model is not currently seen in internal stock."
    return len(base_stock), len(exact_stock), note


def get_deductions(tyre_cond, paint_cond, mech_cond, color_appeal, interior_cond, accidental_repair, service_gap, electrical_work):
    deductions = 0
    if "Average" in tyre_cond:
        deductions += 15000
    elif "Replacement" in tyre_cond:
        deductions += 30000
    if "Minor Scratches" in paint_cond:
        deductions += 15000
    elif "Major Dents" in paint_cond:
        deductions += 40000
    if "Minor Issues" in mech_cond:
        deductions += 20000
    elif "Major Work" in mech_cond:
        deductions += 50000
    if "Low/Unpopular" in color_appeal:
        deductions += 25000
    if interior_cond:
        deductions += 10000
    if accidental_repair:
        deductions += 35000
    if service_gap:
        deductions += 15000
    if electrical_work:
        deductions += 20000
    return deductions


def compute_synthetic_market_price(est_new_price, year, current_km, market_status, brand):
    if est_new_price <= 0 or year == "Any Year" or year is None:
        return 0.0, 0.0, 0.0, "Segment-Aware Synthetic Model"

    age = max(0, CURRENT_YEAR - int(year))
    km_reference = current_km if current_km > 0 else age * 12000
    luxury_brands = {"bmw", "mercedes benz", "mercedes", "audi", "volvo", "jaguar", "land rover", "lexus", "mini", "porsche", "jeep"}

    depreciation = 0.12 + (age * 0.07)
    if normalize_text(brand) in luxury_brands:
        depreciation += 0.05
    if normalize_text(market_status) == "discontinued":
        depreciation += 0.04
    if km_reference > age * 12000 and age > 0:
        depreciation += min(((km_reference - age * 12000) / 10000) * 0.01, 0.06)

    depreciation = min(max(depreciation, 0.20), 0.82)
    price = est_new_price * (1 - depreciation)
    return float(price), float(age), float(km_reference), "Segment-Aware Synthetic Model"


def compute_market_valuation(comparable_pool, weighted_pool, catalog_price, year, km, market_status, brand):
    val = {
        "retail_market_price": 0, "retail_price_low": 0, "retail_price_high": 0,
        "is_synthetic": True, "price_method": "Unknown", "depreciation_percent": 0.0,
        "comps_used": len(comparable_pool), "exact_comps_used": len(weighted_pool), 
        "pricing_scope": "None", "confidence_score": 0, "confidence_label": "Low"
    }
    
    est_new_price = catalog_price
    best_pool = weighted_pool if len(weighted_pool) >= 3 else comparable_pool
    
    if len(best_pool) >= 2:
        val["is_synthetic"] = False
        val["pricing_scope"] = best_pool['pricing_scope'].mode()[0] if not best_pool.empty else "Broad Match"
        val["price_method"] = "Live Market Analysis"
        
        avg_price = best_pool['Price_Raw'].mean()
        std_dev = best_pool['Price_Raw'].std() if len(best_pool) > 2 else (avg_price * 0.05)
        
        val["retail_market_price"] = avg_price
        val["retail_price_low"] = max(0, avg_price - std_dev)
        val["retail_price_high"] = avg_price + std_dev
        
        if est_new_price == 0:
            est_new_price = avg_price * 1.4
            
        val["depreciation_percent"] = max(0, ((est_new_price - avg_price) / est_new_price) * 100) if est_new_price > 0 else 0
        
        km_coverage = comparable_pool["Kilometer"].notna().mean() if not comparable_pool.empty else 0
        owner_coverage = comparable_pool["Owner"].notna().mean() if not comparable_pool.empty and "Owner" in comparable_pool.columns else 0
        source_count = comparable_pool["Source"].nunique() if "Source" in comparable_pool.columns else 1
        
        score, label = compute_confidence_score(val["exact_comps_used"], val["comps_used"], km_coverage, owner_coverage, source_count, False)
        val["confidence_score"] = score
        val["confidence_label"] = label
        
        return val, est_new_price

    if est_new_price > 0 and year is not None and year != "Any Year":
        synth_price, age, km_ref, method = compute_synthetic_market_price(est_new_price, year, km, market_status, brand)
        val["retail_market_price"] = synth_price
        val["retail_price_low"] = synth_price * 0.92
        val["retail_price_high"] = synth_price * 1.08
        val["price_method"] = method
        val["depreciation_percent"] = max(0, ((est_new_price - synth_price) / est_new_price) * 100) if est_new_price > 0 else 0
        val["pricing_scope"] = "Calculated from Master Catalog"
        
        score, label = compute_confidence_score(0, 0, 0, 0, 0, True)
        val["confidence_score"] = score
        val["confidence_label"] = label
        
    return val, est_new_price


def compute_procurement_metrics(retail_price, deductions, margin_pct, demand_score, stock_count, owners):
    refurb_cost = max(deductions, 15000) if retail_price > 0 else 0
    post_refurb_retail = retail_price 
    
    margin_rupees = post_refurb_retail * (margin_pct / 100)
    target_buy_price = max(0, post_refurb_retail - margin_rupees - refurb_cost)
    
    risk_buffer = 1.0
    if demand_score < 40: risk_buffer -= 0.05 
    if stock_count > 2: risk_buffer -= 0.05 
    if type(owners) in [int, float] and owners > 2: risk_buffer -= 0.08 
    
    walkaway_price = target_buy_price * (1 + (0.05 * risk_buffer)) 
    
    return {
        "refurb_cost": refurb_cost,
        "post_refurb_retail": post_refurb_retail,
        "target_buy_price": target_buy_price,
        "walkaway_price": walkaway_price
    }


def evaluate_procurement_decision(valuation, procurement, seller_asking, demand_score, exact_stock_count):
    reasons = []
    decision = "Manual Review"
    decision_color = "warning"

    if valuation["is_synthetic"]:
        reasons.append("Synthetic fallback pricing is active.")
    if valuation["comps_used"] < 3:
        reasons.append(f"Only {valuation['comps_used']} comparable listing(s) available.")
    if valuation["exact_comps_used"] < 2 and not valuation["is_synthetic"]:
        reasons.append("Too few exact same-car comparables.")
    if valuation["confidence_label"] == "Low":
        reasons.append("Confidence score is low.")
    if demand_score < 45:
        reasons.append("Model demand looks slow.")
    if exact_stock_count > 0:
        reasons.append(f"Internal stock already has {exact_stock_count} near-identical car(s).")

    trust_gate_passed = (
        not valuation["is_synthetic"]
        and valuation["comps_used"] >= 3
        and valuation["exact_comps_used"] >= 2
        and valuation["confidence_score"] >= 50
    )

    if seller_asking <= 0:
        reasons.append("Seller asking price not entered yet.")
        return {
            "decision": "Awaiting Ask Price",
            "decision_color": "info",
            "trust_gate_passed": trust_gate_passed,
            "reasons": reasons,
        }

    target_buy_price = procurement["target_buy_price"]
    walkaway_price = procurement["walkaway_price"]

    if not trust_gate_passed:
        if seller_asking > walkaway_price:
            decision = "Reject"
            decision_color = "error"
            reasons.append("Asking price is above walk-away despite weak data confidence.")
        else:
            decision = "Manual Review"
            decision_color = "warning"
            reasons.append("Data is not strong enough for automatic buy approval.")
        return {
            "decision": decision,
            "decision_color": decision_color,
            "trust_gate_passed": trust_gate_passed,
            "reasons": reasons,
        }

    if seller_asking <= target_buy_price:
        decision = "Approve Buy"
        decision_color = "success"
        reasons.append("Asking price is within disciplined target buy price.")
    elif seller_asking <= walkaway_price:
        decision = "Negotiate"
        decision_color = "warning"
        reasons.append("Asking price is above target but still within negotiable range.")
    else:
        decision = "Reject"
        decision_color = "error"
        reasons.append("Asking price is above walk-away price.")

    return {
        "decision": decision,
        "decision_color": decision_color,
        "trust_gate_passed": trust_gate_passed,
        "reasons": reasons,
    }