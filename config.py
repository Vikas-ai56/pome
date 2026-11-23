"""Shared design of the experiment: goods, prices, personas, scenarios."""
import numpy as np

GOODS = ["staples", "milk", "eating_out", "ride", "entertainment"]
GOOD_LABEL = {
    "staples": "Staples (rice, atta, pulses), per kg",
    "milk": "Milk, per litre",
    "eating_out": "Eating out / food delivery, per meal",
    "ride": "Cab or auto ride, per trip",
    "entertainment": "Entertainment (movie, outing), per outing",
}
GOOD_SHORT = {"staples": "Staples", "milk": "Milk", "eating_out": "Eating out",
              "ride": "Rides", "entertainment": "Entertainment"}
BASE_PRICES = {"staples": 60, "milk": 60, "eating_out": 250, "ride": 150, "entertainment": 400}

# Monthly budget (Rs) that the persona allocates across the five goods.
PERSONAS = {
    "low": dict(
        budget=6000,
        desc="a 24-year-old hostel resident on a tight monthly allowance from family, "
             "from a low-income household in a small town, studying in Bengaluru"),
    "lower_middle": dict(
        budget=12000,
        desc="a 30-year-old salaried employee in Bengaluru from a lower-middle-income "
             "family, living in a rented room and sending some money home"),
    "middle": dict(
        budget=25000,
        desc="a 32-year-old software professional in Bengaluru from a middle-income "
             "household, living in a rented one-bedroom flat"),
    "high": dict(
        budget=60000,
        desc="a 38-year-old senior manager in Bengaluru from a high-income household, "
             "living in an owned apartment"),
}

OWN_MULTS = [0.5, 0.75, 1.5, 2.0]          # plus the baseline (1.0)
INCOME_MULTS = [0.5, 0.75, 1.5, 2.0, 3.0]  # plus the baseline (1.0)
N_GARP = 20                                 # random budget sets per consumer
GARP_SEED = 12345


def garp_price_vectors():
    """Fixed set of random price vectors shared by every persona/model/rep."""
    rng = np.random.default_rng(GARP_SEED)
    out = []
    for _ in range(N_GARP):
        m = np.clip(np.exp(rng.normal(0, 0.45, len(GOODS))), 0.4, 2.5)
        out.append({g: int(round(BASE_PRICES[g] * mi)) for g, mi in zip(GOODS, m)})
    return out


def scenarios(reps):
    """Every (persona, scenario, rep) cell of the design."""
    base = dict(BASE_PRICES)
    cells = []
    for persona, pd_ in PERSONAS.items():
        B = pd_["budget"]
        for rep in range(reps):
            cells.append(dict(persona=persona, type="base", sid="base", rep=rep,
                              prices=base, budget=B, g=None, mult=1.0, bmult=1.0))
            for g in GOODS:
                for m in OWN_MULTS:
                    p = dict(base); p[g] = int(round(base[g] * m))
                    cells.append(dict(persona=persona, type="own", sid=f"own_{g}_{m}", rep=rep,
                                      prices=p, budget=B, g=g, mult=m, bmult=1.0))
            for bm in INCOME_MULTS:
                cells.append(dict(persona=persona, type="income", sid=f"inc_{bm}", rep=rep,
                                  prices=base, budget=int(round(B * bm)), g=None, mult=1.0, bmult=bm))
            for k, p in enumerate(garp_price_vectors()):
                cells.append(dict(persona=persona, type="garp", sid=f"garp_{k}", rep=rep,
                                  prices=p, budget=B, g=None, mult=None, bmult=1.0))
    return cells


def build_prompt(persona, prices, budget):
    lines = "\n".join(f"- {GOOD_LABEL[g]}: Rs {prices[g]}" for g in GOODS)
    keys = ", ".join(f'"{g}": <number>' for g in GOODS)
    return (
        "You are role-playing a consumer in urban India (Bengaluru).\n"
        f"Profile: {PERSONAS[persona]['desc']}.\n\n"
        f"For the coming month you have a fixed budget of Rs {budget:,} to spend only on the five "
        "goods below. Decide how many units of each good to buy. You do not have to buy every "
        "good, quantities may be decimals, and any money you do not spend is lost, so spend "
        "your whole budget.\n\n"
        f"Prices:\n{lines}\n\n"
        "Reply with ONLY a JSON object of the form {" + keys + "} giving the quantity of each "
        "good you choose. No explanation."
    )
