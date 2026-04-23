# Acme Cloud — Billing FAQ

## How is my bill calculated?

Acme Cloud bills monthly in arrears. Your bill is the sum of:
- Your base plan fee (Starter €9, Growth €49, Scale €129, Enterprise €489)
- Any overage charges (extra compute, bandwidth, storage beyond plan limits)
- Any add-ons (managed Postgres, Redis, custom domains, SSL certificates)

Bills are issued on the 1st of each month covering the prior month, and due within 14 days.

## What's on my Growth plan?

Growth includes: 4 vCPU, 16 GB RAM, 200 GB SSD, 2 TB egress, 1 managed Postgres instance, 24/5 email support. Overage: €0.05 per GB egress, €0.01 per GB-hour storage over 200 GB.

## How do I change plans?

Plan changes take effect immediately. Upgrades are prorated from today through the end of the billing period. Downgrades apply from the next billing period to avoid refund complexity. You can change plans from the dashboard at `cloud.acme.com/billing/plan` or by asking voice support.

## What happens if I don't pay?

Bills overdue by 7 days trigger an email reminder. At 14 days overdue, new resources cannot be provisioned. At 21 days, services are suspended (data retained, not deleted). At 60 days, accounts are cancelled and data is permanently deleted after a 30-day grace period.

## How do I get a refund?

Refunds are issued for duplicate charges, failed service-level agreement commitments, or cancelled add-ons within 7 days of purchase. Contact support with your invoice number.

## Tax

EU VAT is applied based on the billing address on file. Business accounts with a valid VAT ID receive reverse-charge treatment.
