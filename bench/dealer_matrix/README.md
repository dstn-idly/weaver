# Vehicle-v2 compatibility sample

This is a read-only compatibility sample, not a production crawl. It tests one bounded inventory/listing page (SRP) and one representative vehicle detail page (VDP) per dealership. It does not prove full inventory coverage, customer ownership/authorization, or production readiness.

The latest completed artifact is [compatibility-20260825T033214Z.json](results/compatibility-20260825T033214Z.json).

## Exact run totals

| bucket | count |
| --- | ---: |
| sites requested/run | 20 / 20 |
| `sample_pass` | 0 |
| `sample_partial` | 0 |
| `candidate_failure` | 18 |
| `blocked` | 2 |
| robots.txt requests | 0 |

`candidate_failure` means the application-generated selector contract could not prove a VIN-scoped detail root/gallery/item combination, so Luna was intentionally not called for that site. `blocked` means the isolated site worker exceeded its 60-second wall-clock bound. No result is being treated as a full-inventory success.

## Site/platform breakdown

| site | expected platform | result | reason |
| --- | --- | --- | --- |
| orlandonissan-com | Team Velocity (Apollo) | candidate_failure | no verified detail root/gallery/item contract |
| universal-nissan-com | Dealer Inspire (Cars Commerce) | candidate_failure | no verified detail root/gallery/item contract |
| serramonteford-com | Dealer.com (Cox) | candidate_failure | no verified detail root/gallery/item contract |
| northshoremitsubishi-ca | EDealer | blocked | 60s worker bound |
| ricart-com | DealerOn | candidate_failure | no verified detail root/gallery/item contract |
| huberautomotive-com | Dealer.com (Cox) | candidate_failure | no verified detail root/gallery/item contract |
| bobrossbuickgmc-com | Dealer Inspire (Cars Commerce) | candidate_failure | no verified detail root/gallery/item contract |
| mattcastrucciautomall-com | Dealer eProcess | candidate_failure | no verified detail root/gallery/item contract |
| olympicautoga-com | DealerCenter / DWS | candidate_failure | no verified detail root/gallery/item contract |
| marhofer-com | Remora | candidate_failure | no verified detail root/gallery/item contract |
| wylereastgate-com | Motive (ridemotive.com) | candidate_failure | no verified detail root/gallery/item contract |
| automania-us | VehiclesNETWORK (apogeeINVENT) | candidate_failure | no verified detail root/gallery/item contract |
| rideplaza-com | unidentified | candidate_failure | no verified detail root/gallery/item contract |
| vickar-com | Convertus | candidate_failure | no verified detail root/gallery/item contract |
| autoshowwinnipeg-com | D2C Media | blocked | 60s worker bound |
| 401dixiekia-com | SM360 | candidate_failure | no verified detail root/gallery/item contract |
| birchwood-ca | unidentified (in-house WordPress) | candidate_failure | no verified detail root/gallery/item contract |
| centreautomobilesduquebec-com | ADWS | candidate_failure | no verified detail root/gallery/item contract |
| occasionbeaucage-com | Autoroot Technologies | candidate_failure | no verified detail root/gallery/item contract |
| ridetime-ca | unidentified (WordPress) | candidate_failure | no verified detail root/gallery/item contract |

## Acceptance contract

A site can become `sample_pass` only when the local replay proves all of the following for the representative VDP:

- real VIN identity bound to the page-primary vehicle;
- at least two unique VDP-owned gallery URLs;
- numeric full-resolution evidence with width at least 600;
- replayed sample fields: VIN, year, make/model or name, price, mileage, distance unit, color, description, and detail URL.

The script uses the vehicle transport directly for this public compatibility probe. It never consults or requests `robots.txt`, does not request owner authorization, and does not persist HTML, cookies, browser profiles, or secrets. Luna is server-side only and is invoked only after application-generated card/detail/gallery candidates pass local contract checks. Per-site workers run in isolated process groups, at most two concurrently, with a 20-second maximum inference request and a 60-second total worker deadline.

Run from the repository root with the server-side environment loaded:

```sh
set -a; source .env; set +a
./.venv/bin/python bench/dealer_matrix_compat.py --max-sites 20
```
