import numpy as np
import pandas as pd

# ASSUMPTION: weekday demand shape Mon..Sun (weekend peak, Fri/Sat highest)
WEEKDAY = [0.90, 0.85, 0.90, 1.00, 1.20, 1.40, 1.15]

# ASSUMPTION per category: (shelf_days, unit_cost, price_markup, avg_daily_demand, elasticity)
CATS = {
    "Bakery":      (3, 1.0, 1.8, 40, 2.2),
    "Dairy":       (7, 1.2, 1.4, 35, 1.5),
    "Produce":     (5, 0.8, 1.6, 45, 1.8),
    "Meat":        (4, 3.5, 1.35, 20, 1.6),
    "Ready meals": (3, 2.5, 1.5, 18, 2.0),
}

def make_products(seed=7):
    """20 synthetic products: 4 per category, with random variation."""
    rng = np.random.default_rng(seed)
    rows = []
    for cat, (shelf, cost, mk, dem, el) in CATS.items():
        for i in range(4):
            c = round(cost * rng.uniform(0.8, 1.2), 2)
            rows.append(dict(product=f"{cat} {i+1}", category=cat, shelf=shelf,
                             cost=c, price=round(c * mk, 2),
                             demand=round(dem * rng.uniform(0.7, 1.3)),
                             elast=round(el * rng.uniform(0.85, 1.15), 2)))
    return pd.DataFrame(rows)

def simulate(prods, rules, days=30, seed=42, response=1.0, order_mult=1.25):
    """
    rules: {days_left_threshold: discount}, e.g. {2: 0.2, 1: 0.4}. {} = fixed price.
    Model:
      - One batch arrives per product per day: qty = demand * order_mult (overstock -> waste).
      - Batch expires after `shelf` days; unsold units at expiry are wasted.
      - Daily demand ~ Poisson(base * weekday * (1 - best_discount)^(-elast * response)).
      - Shoppers buy oldest batch first, paying that batch's (discounted) price.
      - Margin = revenue - unit_cost * (units sold + units wasted).
    """
    rng = np.random.default_rng(seed)
    sales, summ = [], []
    for _, p in prods.iterrows():
        batches, wasted, ordered = [], 0, 0
        for d in range(days):
            q = int(round(p.demand * order_mult))
            batches.append([d + p.shelf, q]); ordered += q
            batches.sort()
            def disc(b):
                left = b[0] - d
                ds = [v for k, v in rules.items() if left <= k]
                return max(ds) if ds else 0.0
            best = max(disc(b) for b in batches)
            lam = p.demand * WEEKDAY[d % 7] * (1 - best) ** (-p.elast * response)
            want = rng.poisson(lam)
            for b in batches:
                if want <= 0: break
                take = min(want, b[1]); want -= take; b[1] -= take
                if take:
                    dc = disc(b)
                    sales.append(dict(product=p["product"], category=p.category, day=d,
                                      discount=dc, units=take,
                                      revenue=take * p.price * (1 - dc)))
            for b in batches:                    # expire at end of last sellable day
                if b[0] == d + 1: wasted += b[1]
            batches = [b for b in batches if b[0] > d + 1]
        summ.append(dict(product=p["product"], category=p.category, ordered=ordered, wasted=wasted,
                         cogs_per=p.cost))
    s = pd.DataFrame(sales); m = pd.DataFrame(summ)
    rev = s.groupby("product").revenue.sum(); sold = s.groupby("product").units.sum()
    m["sold"] = m["product"].map(sold).fillna(0)
    m["revenue"] = m["product"].map(rev).fillna(0)
    m["margin"] = m.revenue - m.cogs_per * (m.sold + m.wasted)
    return s, m

def compare(prods, rules, seeds, response=1.0):
    out = []
    for sd in seeds:
        _, a = simulate(prods, {}, seed=sd, response=response)
        _, b = simulate(prods, rules, seed=sd, response=response)
        out.append(dict(seed=sd,
                        waste_reduction_pct=100 * (a.wasted.sum() - b.wasted.sum()) / a.wasted.sum(),
                        margin_change_pct=100 * (b.margin.sum() - a.margin.sum()) / abs(a.margin.sum())))
    return pd.DataFrame(out)