# Angel One SmartAPI Authentication Guide

This document details the authentication architecture, token management, security standards, and local smoke testing for the Angel One SmartAPI integration in `adaptive-algo-trading-platform`.

---

## 1. Purpose of the SmartAPI Integration

Angel One's SmartAPI provides the primary broker interface for Indian equities and derivatives (NSE / BSE). In this step, we implement a dedicated, isolated authentication and session-management layer that:
- Generates RFC 6238 Time-based One-Time Passwords (TOTP) on demand.
- Obtains authorized session credentials via `SmartConnect.generateSession()`.
- Distinguishes between REST API authorization tokens and WebSocket streaming tokens.
- Protects credentials and token payloads from ever appearing in logs, version control, or error messages.

---

## 2. Required Credentials & `.env` Setup

Add the following environment variables to your local `.env` file (never commit this file):

```env
# Angel One SmartAPI Credentials
ANGELONE_API_KEY=your_smartapi_api_key_here
ANGELONE_CLIENT_CODE=your_client_id_here
ANGELONE_PASSWORD=your_account_password_or_mpin_here
ANGELONE_TOTP_SECRET=your_base32_totp_secret_here
```

### Where to Find Your Credentials:
1. **API Key**: Generated on the [Angel One SmartAPI Developer Portal](https://smartapi.angelbroking.com/).
2. **Client Code**: Your 6-8 character Angel One account username/client ID.
3. **Password**: Your login password or PIN.
4. **TOTP Secret**: The Base32 alphanumeric secret key displayed when enabling TOTP / 2FA in your Angel One profile (or QR code text).

---

## 3. Authentication Architecture Flow

```text
.env (Local Environment)
        ↓
AngelOneSettings (Pydantic Configuration)
        ↓
pyotp.TOTP (RFC 6238 6-Digit Code)
        ↓
SmartConnect.generateSession(clientCode, password, totp)
        ↓
AngelOneSession (In-Memory Container)
├── jwt_token       ──► REST API Authorization (Historical & Orders)
├── refresh_token   ──► Session Renewal
└── feed_token      ──► WebSocket Streaming (Live Ticks & Quotes)
```

---

## 4. Token Roles & Guidance

| Token Name | SmartAPI Field | Purpose in Platform |
| :--- | :--- | :--- |
| **JWT Token** | `jwtToken` | Passed in HTTP headers (`Authorization: Bearer <jwtToken>`) for all REST API endpoints (e.g. historical data fetching). |
| **Refresh Token** | `refreshToken` | Used to renew the session when the JWT expires without needing full password + TOTP re-entry. |
| **Feed Token** | `feedToken` | Used exclusively for WebSocket connections to stream live ticks and order book updates. |

---

## 5. Security & Secret Protection Standards

1. **Zero Secret Logging**: The `AngelOneSession` class automatically masks token strings in `__repr__()`. Loggers output only status strings and timestamps.
2. **No Database Token Persistence**: Active tokens reside strictly in memory for the lifecycle of the client process.
3. **Git Protection**: `.gitignore` explicitly prevents `.env` and local model/session artifacts from being tracked.
4. **Clean Error Messages**: Exceptions never print credentials or secret values in error traces.

---

## 6. How to Run the Local Authentication Smoke Test

Once you have configured your `.env` file with valid credentials, run the authentication smoke test from your terminal:

```bash
# Run the Angel One authentication CLI smoke test
python -m adaptive_trading.integrations.angel_one.cli
```

### Expected Output on Success:
```text
============================================================
      ANGEL ONE SMARTAPI AUTHENTICATION SMOKE TEST          
============================================================
Client Code       : A123456
Authenticated At  : 2026-08-30T14:50:00.000000+00:00
REST Authorization: Active (JWT Token generated)
WebSocket Stream  : Active (Feed Token generated)
------------------------------------------------------------
Angel One authentication successful.
Session established successfully.
============================================================
```

---

## 7. What This Step Does NOT Implement

This step is strictly limited to authentication. It does **not**:
- Download historical candle data (`getCandleData` is implemented in Step 11)
- Subscribe to WebSockets or live market feeds
- Place, modify, or cancel orders
- Access live portfolio positions or funds
- Implement trading strategies or backtests

