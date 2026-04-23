# Acme Cloud — Regions

Acme Cloud operates data centers across Europe:

| Region code | Location            | Availability zones | Notes |
|-------------|---------------------|--------------------|-------|
| fra-1       | Frankfurt, Germany  | a, b, c            | Flagship region, all services available |
| ams-1       | Amsterdam, NL       | a, b               | All services available |
| par-1       | Paris, France       | a, b               | All services available |
| waw-1       | Warsaw, Poland      | a                  | Compute + Postgres; Redis coming 2026-Q3 |
| dub-1       | Dublin, Ireland     | a, b               | Compute + Postgres; expanding |

## Choosing a region

Pick the region closest to your primary user base to minimize latency. Cross-region egress is billed as external egress. For EU-citizen data, any of the above regions keep data inside the EU.

## Changing regions

Region migration is not automatic. You can provision new resources in a target region and migrate data (Postgres has built-in logical replication). Contact support for assistance with zero-downtime migrations on Scale and Enterprise plans.
