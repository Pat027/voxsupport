# Acme Cloud — Managed Postgres

## Creating an instance

Managed Postgres is included on Growth plans and above. Create an instance from the dashboard at `cloud.acme.com/databases` or via CLI:

```
acme db create --name mydb --region fra-1 --plan growth
```

Instances provision in 60-90 seconds. Connection strings are shown in the dashboard.

## Connection issues

Most connection failures fall into four categories:

**1. Firewall / allow-list.** By default, managed Postgres only accepts connections from within the same project. To connect from an external IP, add it to the allow-list under `Database → Network`.

**2. Credentials rotated.** If you recently regenerated credentials, any cached connection string is invalid. Use the new connection string shown in the dashboard.

**3. Instance in failover.** During failover (triggered by infrastructure events or version upgrades), connections are interrupted for 20-60 seconds. The dashboard shows "Failing over" in the instance card. Retry after the dashboard reports "Healthy".

**4. Resource exhaustion.** If your instance is hitting connection limits (50 on Growth, 200 on Scale), new connections are refused. Scale up the plan, use pgBouncer (managed pgBouncer available on Scale and above), or reduce idle connections.

## Automatic failover

Managed Postgres on Scale and Enterprise runs with a hot standby. Failover is automatic on primary failure, with typical RTO of 45 seconds and RPO of zero (synchronous replication).

## Backups

Daily backups retained for 14 days (Growth), 30 days (Scale), 90 days (Enterprise). Point-in-time recovery available within the retention window.

## Supported versions

Postgres 14, 15, 16. Version 16 is default for new instances. Upgrades are in-place with a brief failover.
