#!/usr/bin/env python3
"""
Reproducible factorization table of the Mersenne spectrum M_eps = 2^eps - 1,
separating the two growth modes of the pending-sign recursion:
  - INNOVATION:     M_eps introduces a prime not seen at any lower depth
  - CRYSTALLIZATION: M_eps contains a repeated prime (I_or > 0)
and computing the nested-pending-sign count I_or(eps) = sum_i (alpha_i - 1).

Self-contained; standard library only. Run: python3 mersenne_spectrum.py
"""

def factorize(n):
    """Return dict {prime: multiplicity} for n >= 2."""
    f = {}
    d = 2
    m = n
    while d * d <= m:
        while m % d == 0:
            f[d] = f.get(d, 0) + 1
            m //= d
        d += 1
    if m > 1:
        f[m] = f.get(m, 0) + 1
    return f

def I_or(eps):
    """Nested pending-sign count: sum over prime multiplicities of (alpha - 1)."""
    M = 2**eps - 1
    if M <= 1:
        return 0
    return sum(a - 1 for a in factorize(M).values())

def spectrum_table(eps_max):
    """Build the full table with both growth-mode flags."""
    rows = []
    seen_primes = set()
    for eps in range(1, eps_max + 1):
        M = 2**eps - 1
        fac = factorize(M) if M > 1 else {}
        primes = set(fac.keys())
        new_primes = primes - seen_primes
        repeated = any(a > 1 for a in fac.values())
        rows.append({
            "eps": eps,
            "M": M,
            "factorization": fac,
            "I_or": sum(a - 1 for a in fac.values()) if M > 1 else 0,
            "innovation": len(new_primes) > 0,      # introduces a new prime
            "new_primes": sorted(new_primes),
            "crystallization": repeated,             # repeats a prime (I_or > 0)
            "pure_crystallization": repeated and len(new_primes) == 0,
        })
        seen_primes |= primes
    return rows

def fmt_fac(fac):
    if not fac:
        return "1"
    return " * ".join(f"{p}^{a}" if a > 1 else f"{p}" for p, a in sorted(fac.items()))

if __name__ == "__main__":
    rows = spectrum_table(24)
    header = f"{'eps':>3} | {'M_eps':>12} | {'factorization':<26} | {'I_or':>4} | {'innov':>5} | {'cryst':>5} | new primes"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['eps']:>3} | {r['M']:>12} | {fmt_fac(r['factorization']):<26} | "
              f"{r['I_or']:>4} | {'yes' if r['innovation'] else ' no':>5} | "
              f"{'YES' if r['crystallization'] else ' no':>5} | "
              f"{','.join(map(str, r['new_primes'])) if r['new_primes'] else '(none)'}")
    innov = [r['eps'] for r in rows if r['innovation']]
    cryst = [r['eps'] for r in rows if r['crystallization']]
    pure  = [r['eps'] for r in rows if r['pure_crystallization']]
    print("\nInnovation levels (introduce a new prime):", innov)
    print("Crystallization levels (I_or > 0, repeated prime):", cryst)
    print("Pure-crystallization levels (repeat, no new prime):", pure)
    print("I_or by level:", {r['eps']: r['I_or'] for r in rows if r['I_or'] > 0})
