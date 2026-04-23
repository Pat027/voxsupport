# Acme Cloud — Voice Support Scope

## What voice support can help with

- Account status, plan details, billing questions, invoice explanations
- Plan upgrades and downgrades
- Common technical issues on managed services (Postgres, Redis, compute instances)
- Lookup of ticket status and escalation
- Account cancellation requests

## What voice support cannot do

Voice support cannot:

- **Execute destructive operations**: delete databases, delete accounts, trigger mass deletes. These require confirmation in the dashboard with a typed confirmation string.
- **Reveal secrets**: API keys, database passwords, connection strings. Secrets are shown only in the dashboard.
- **Change security settings**: rotate credentials, modify firewall rules, grant team access. These require dashboard actions for audit trail.
- **Discuss other customers' data**: agent has no access to accounts other than the authenticated caller's.
- **Provide legal, tax, or financial advice**: for regulatory questions about VAT, contracts, or compliance attestations, voice support transfers to the relevant team.

## Authentication

Voice support verifies callers by email lookup + one confirming data point (last 4 of latest invoice, or region of the primary instance). If verification fails twice, the call is transferred to a human agent.

## Escalation triggers

The agent explicitly escalates to a human when:

1. The caller requests a human ("transfer me", "let me speak to someone")
2. The detected intent is outside the supported scope
3. Authentication fails
4. The agent's confidence in its own answer falls below the configured threshold
5. The customer is on an Enterprise plan and requests their dedicated account manager
