# TradingView webhook -> paper trader

Replaces the screen-reading watcher. TradingView tells the paper trader about every strategy fill itself, so
nothing is read off a screen and nothing is missed while the PC is busy.

## What you need
- A TradingView plan that allows webhook alerts (the alert dialog has a **Webhook URL** box under Notifications).
- The paper trader's PC on and the trader running (Start trading) - the receiver writes straight into `livepaper.db`.
- A public address for the receiver. Easiest: Cloudflare's free quick tunnel (no account):
  `cloudflared tunnel --url http://localhost:8787` prints `https://something.trycloudflare.com`. That address changes
  every time you restart the tunnel, so update the alert when it does. A named tunnel or a NAS keeps it fixed.

## 1. Start the receiver
```
.venv\Scripts\python.exe tools\tv_webhook.py --token CHOOSE-A-LONG-SECRET --dry-run
```
Dry run first: it logs what it would do and sends nothing. Drop `--dry-run` to go live. Keep it running.

## 2. Make the alert (TradingView)
1. Open the GRIND BTC [45MIN] chart, press **Alt+A** (create alert).
2. Condition: **GRIND BTC [45MIN]**, then **Order fills only**.
3. Notifications: tick **Webhook URL** and paste `https://your-address/` .
4. Message (replace the secret with yours):
```
{"token":"CHOOSE-A-LONG-SECRET","position":"{{strategy.market_position}}","size":{{strategy.market_position_size}},"contracts":{{strategy.order.contracts}},"price":{{strategy.order.price}},"time":"{{timenow}}"}
```
5. Create. Webhooks are sent only on bar close/fill, so there can be a short delay.

## What it does
- Position grows (new entry or pyramid add): opens a paper run `TV-<month day>-<time>` at TradingView's fill price, sized
  from TradingView's order size, with TradingView's ladder (stop 9%, TP 1.7 / 2.3 / 3.4% = 33 / 33 / 34%).
- Position goes flat: closes every run it opened, at TradingView's price.
- Partial exit: ignored.
- Every alert and action is appended to `runtime\tv_webhook_log.csv`.

Open `https://your-address/` in a browser: it should say "tv_webhook is running". A wrong token gets a 403.

## More than one chart
Each chart gets its own alert and its own `"chart"` name in the message, so positions never mix. The BTC 30m chart needs
no change. For ETH on the 1d chart, add three fields to its message:
```
{"token":"...","chart":"ETH1D","symbol":"ETHUSD","timeframe":"1d","position":"{{strategy.market_position}}","size":{{strategy.market_position_size}},"contracts":{{strategy.order.contracts}},"price":{{strategy.order.price}},"time":"{{timenow}}"}
```
Runs are named `TV-ETH1D-<date>-<time>`, keep the same stop and targets and the same limit of 8, and the position is kept in
`runtime\tv_webhook_state_ETH1D.json`. A flat alert on one chart only closes that chart's runs.
