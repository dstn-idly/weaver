import json

from weaver.vehicle.models import DetailSpec, FieldRule
from weaver.vehicle.vdp import extract_vdp


VIN = "1HGBH41JXMN109186"
URL = f"https://dealer.example/vdp/{VIN}"


def test_vdp_requires_page_primary_identity_and_owns_gallery() -> None:
    html = f"""
    <html><head><link rel="canonical" href="{URL}"></head><body>
      <main class="vehicle" data-vin="{VIN}">
        <h1 class="title">2025 Honda Civic</h1>
        <section class="primary-gallery">
          <a href="/photos/{VIN}-front.jpg"><img data-full="/photos/{VIN}-front.jpg" width="1600"></a>
          <img data-full="/photos/{VIN}-side.jpg" width="1600">
        </section>
        <aside class="related-vehicles"><img src="/photos/other-car.jpg"></aside>
      </main>
    </body></html>
    """
    detail = DetailSpec(
        root_selector="main.vehicle",
        gallery_selector=".primary-gallery",
        gallery_item_selector="img",
        fields={
            "vin": FieldRule("[data-vin]", "data-vin", "vin"),
            "name": FieldRule(".title"),
        },
        max_photos=80,
    )
    result = extract_vdp(html, detail_url=URL, origin="https://dealer.example", detail=detail, expected_vin=VIN)
    assert result.identity_proven
    assert result.record["vin"] == VIN
    assert result.record["photos"] == [
        f"https://dealer.example/photos/{VIN}-front.jpg",
        f"https://dealer.example/photos/{VIN}-side.jpg",
    ]
    assert all("other-car" not in photo.url for photo in result.photos)


def test_vdp_accepts_one_explicit_page_vin_outside_the_gallery_root() -> None:
    """Dealer platforms may keep the canonical VIN in a sibling lead form."""

    html = f"""
    <html><head><link rel="canonical" href="{URL}"></head><body class="vdp">
      <form class="lead-form"><input id="vin-{VIN}" name="vin" value="{VIN}"></form>
      <main class="vehicle">
        <h1 class="title">2025 Honda Civic</h1>
        <section class="primary-gallery">
          <img data-full="/photos/{VIN}-front.jpg" width="1600">
          <img data-full="/photos/{VIN}-side.jpg" width="1600">
        </section>
      </main>
      <aside class="compare-vehicles">
        <input name="vin" value="1M8GDM9AXKP042788">
        <img src="/photos/other-car.jpg">
      </aside>
    </body></html>
    """
    detail = DetailSpec(
        root_selector="main.vehicle",
        gallery_selector=".primary-gallery",
        gallery_item_selector="img",
        fields={"name": FieldRule(".title")},
        max_photos=80,
    )

    result = extract_vdp(html, detail_url=URL, origin="https://dealer.example", detail=detail, expected_vin=VIN)

    assert result.identity_proven
    assert result.record["vin"] == VIN
    assert result.record["photos"] == [
        f"https://dealer.example/photos/{VIN}-front.jpg",
        f"https://dealer.example/photos/{VIN}-side.jpg",
    ]
    assert all("other-car" not in photo.url for photo in result.photos)


def test_vdp_rejects_two_unowned_page_level_vin_signals() -> None:
    other_vin = "1M8GDM9AXKP042788"
    opaque_url = "https://dealer.example/used/vehicle/civic-id123.htm"
    html = f"""
    <html><head><link rel="canonical" href="{opaque_url}"></head><body>
      <input name="vin" value="{VIN}">
      <input name="vin" value="{other_vin}">
      <main class="vehicle">
        <section class="primary-gallery">
          <img data-full="/photos/{VIN}-front.jpg" width="1600">
          <img data-full="/photos/{VIN}-side.jpg" width="1600">
        </section>
      </main>
    </body></html>
    """
    detail = DetailSpec(
        root_selector="main.vehicle",
        gallery_selector=".primary-gallery",
        gallery_item_selector="img",
        fields={},
    )

    result = extract_vdp(html, detail_url=opaque_url, origin="https://dealer.example", detail=detail, expected_vin=VIN)

    assert not result.identity_proven
    assert result.photos == ()


def test_vdp_normalizes_owned_edealer_gallery_to_full_resolution() -> None:
    """A known CDN rendition upgrade must retain VIN/gallery ownership."""

    html = f"""
    <html><head><link rel="canonical" href="{URL}"></head><body class="vdp">
      <input name="vin" value="{VIN}">
      <main class="vehicle">
        <section class="photo-slide-info-row">
          <img data-src="https://images.edealer.ca/2/187170893.jpeg">
          <img data-src="https://images.edealer.ca/2/187170876.jpeg">
        </section>
      </main>
      <aside class="related-vehicles">
        <img data-src="https://images.edealer.ca/2/999999999.jpeg">
      </aside>
    </body></html>
    """
    detail = DetailSpec(
        root_selector="main.vehicle",
        gallery_selector=".photo-slide-info-row",
        gallery_item_selector="img",
        fields={},
    )

    result = extract_vdp(
        html,
        detail_url=URL,
        origin="https://dealer.example",
        detail=detail,
        expected_vin=VIN,
    )

    assert result.identity_proven
    assert result.record["photos"] == [
        "https://images.edealer.ca/0/187170893.jpeg",
        "https://images.edealer.ca/0/187170876.jpeg",
    ]
    assert all(photo.source == "known_cdn_full" for photo in result.photos)
    assert all(photo.width == 1600 for photo in result.photos)
    assert all(photo.full_resolution_candidate for photo in result.photos)
    assert all("999999999" not in photo.url for photo in result.photos)


def test_vdp_does_not_guess_full_resolution_paths_for_unknown_cdns() -> None:
    html = f"""
    <html><head><link rel="canonical" href="{URL}"></head><body class="vdp">
      <input name="vin" value="{VIN}">
      <main class="vehicle">
        <section class="primary-gallery">
          <img data-src="https://cdn.example/2/187170893.jpeg">
          <img data-src="https://cdn.example/2/187170876.jpeg">
        </section>
      </main>
    </body></html>
    """
    detail = DetailSpec(
        root_selector="main.vehicle",
        gallery_selector=".primary-gallery",
        gallery_item_selector="img",
        fields={},
    )

    result = extract_vdp(
        html,
        detail_url=URL,
        origin="https://dealer.example",
        detail=detail,
        expected_vin=VIN,
    )

    assert result.record["photos"] == [
        "https://cdn.example/2/187170893.jpeg",
        "https://cdn.example/2/187170876.jpeg",
    ]
    assert all(photo.source == "lazy_src" for photo in result.photos)
    assert all(photo.width is None for photo in result.photos)
    assert all(not photo.full_resolution_candidate for photo in result.photos)


def test_vdp_normalizes_exact_cai_vehicle_assets_without_changing_identity() -> None:
    first = "830919e5-e5aa-4aba-a8b6-5f4db62a9882.jpg"
    second = "27da855c-e7f7-41cb-b42b-955bb5027837.jpg"
    html = f"""
    <html><head><link rel="canonical" href="{URL}"></head><body class="vdp">
      <input name="vin" value="{VIN}">
      <main class="vehicle">
        <section class="primary-gallery">
          <img src="https://assets.cai-media-management.com/resize/640x640/common-vehicle-media/{first}">
          <img src="https://assets.cai-media-management.com/resize/640x640/common-vehicle-media/{second}">
        </section>
      </main>
      <aside class="related-vehicles">
        <img src="https://assets.cai-media-management.com/resize/640x640/common-vehicle-media/"
             alt="related vehicle">
      </aside>
    </body></html>
    """
    detail = DetailSpec(
        root_selector="main.vehicle",
        gallery_selector=".primary-gallery",
        gallery_item_selector="img",
        fields={},
    )

    result = extract_vdp(
        html,
        detail_url=URL,
        origin="https://dealer.example",
        detail=detail,
        expected_vin=VIN,
    )

    assert result.record["photos"] == [
        f"https://assets.cai-media-management.com/common-vehicle-media/{first}",
        f"https://assets.cai-media-management.com/common-vehicle-media/{second}",
    ]
    assert all(photo.source == "known_cdn_full" for photo in result.photos)
    assert all(photo.full_resolution_candidate for photo in result.photos)


def test_vdp_does_not_rewrite_cai_lookalikes_or_foreign_hosts() -> None:
    good_uuid = "830919e5-e5aa-4aba-a8b6-5f4db62a9882.jpg"
    non_uuid = "jim-norton-rav4-front.jpg"
    foreign = f"https://cdn.example/resize/640x640/common-vehicle-media/{good_uuid}"
    wrong_path = f"https://assets.cai-media-management.com/resize/640x640/related-vehicle-media/{good_uuid}"
    cai_non_uuid = f"https://assets.cai-media-management.com/resize/640x640/common-vehicle-media/{non_uuid}"
    html = f"""
    <html><head><link rel="canonical" href="{URL}"></head><body class="vdp">
      <input name="vin" value="{VIN}">
      <main class="vehicle">
        <section class="primary-gallery">
          <img src="{foreign}">
          <img src="{wrong_path}">
          <img src="{cai_non_uuid}">
        </section>
      </main>
    </body></html>
    """

    result = extract_vdp(
        html,
        detail_url=URL,
        origin="https://dealer.example",
        detail=DetailSpec(
            root_selector="main.vehicle",
            gallery_selector=".primary-gallery",
            gallery_item_selector="img",
            fields={},
        ),
        expected_vin=VIN,
    )

    assert result.record["photos"] == [foreign, wrong_path, cai_non_uuid]
    assert all(photo.source == "img_src" for photo in result.photos)
    assert all(not photo.full_resolution_candidate for photo in result.photos)


def test_vdp_rejects_source_images_coming_placeholder() -> None:
    html = f"""
    <html><head><link rel="canonical" href="{URL}"></head><body class="vdp">
      <input name="vin" value="{VIN}">
      <main class="vehicle">
        <section class="primary-gallery">
          <img src="https://static.edealer.ca/assets/new_vehicles_images_coming.png">
        </section>
      </main>
    </body></html>
    """
    detail = DetailSpec(
        root_selector="main.vehicle",
        gallery_selector=".primary-gallery",
        gallery_item_selector="img",
        fields={},
    )

    result = extract_vdp(
        html,
        detail_url=URL,
        origin="https://dealer.example",
        detail=detail,
        expected_vin=VIN,
    )

    assert result.identity_proven
    assert result.photos == ()
    assert "photo" not in result.record
    assert "photos" not in result.record


def test_vdp_enriches_owned_nested_microdata_and_ignores_related_values() -> None:
    html = f"""
    <html><head>
      <link rel="canonical" href="{URL}">
      <meta name="description" content="One-owner Honda Civic with service history.">
      <meta property="og:description" content="One-owner Honda Civic with service history…">
    </head><body class="vdp">
      <main class="vehicle" data-vin="{VIN}">
        <span itemprop="mileageFromOdometer">
          <meta itemprop="value" content="145444">
          <meta itemprop="unitCode" content="KMT">
        </span>
        <span itemprop="vehicleInteriorColor">Black</span>
        <span itemprop="vehicleTransmission">Automatic</span>
        <span itemprop="driveWheelConfiguration">https://schema.org/4x4Configuration</span>
        <div class="vehicle-features"><span class="vdp-item">Heated seats</span></div>
        <section class="primary-gallery">
          <img data-full="/photos/{VIN}-front.jpg" width="1600">
          <img data-full="/photos/{VIN}-side.jpg" width="1600">
        </section>
      </main>
      <aside class="related-vehicles">
        <span itemprop="vehicleTransmission">Manual</span>
        <span itemprop="mileageFromOdometer"><meta itemprop="value" content="9"></span>
      </aside>
    </body></html>
    """
    detail = DetailSpec(
        root_selector="body",
        gallery_selector=".primary-gallery",
        gallery_item_selector="img",
        fields={},
    )

    result = extract_vdp(
        html,
        detail_url=URL,
        origin="https://dealer.example",
        detail=detail,
        expected_vin=VIN,
    )

    assert result.record["mileage"] == 145444
    assert result.record["distance_unit"] == "km"
    assert result.record["color_int"] == "Black"
    assert result.record["transmission"] == "Automatic"
    assert result.record["drivetrain"] == "4x4"
    assert result.record["features"] == ["Heated seats"]
    assert result.record["description"] == "One-owner Honda Civic with service history."


def test_vdp_accepts_vin_bound_custom_gallery_and_same_vin_canonical_alias() -> None:
    requested = f"https://dealer.example/viewdetails/used/{VIN}/2025-honda-civic"
    canonical = f"https://dealer.example/viewdetails/cpo/{VIN}/2025-honda-civic"
    other_vin = "1M8GDM9AXKP042788"
    html = f"""
    <html><head><link rel="canonical" href="{canonical}"></head><body>
      <main class="vehicle-detail">
        <oem-gallery-component
          :vin="'{VIN}'"
          :photoUrls="'https://content.homenetiol.com/2000157/2065512/0x0/front.jpg,https://content.homenetiol.com/2000157/2065512/0x0/side.jpg,https://content.homenetiol.com/2000157/2065512/0x0/rear.jpg'">
        </oem-gallery-component>
      </main>
      <aside class="related-vehicles">
        <oem-gallery-component
          :vin="'{other_vin}'"
          :photoUrls="'https://content.homenetiol.com/9/9/0x0/other-front.jpg,https://content.homenetiol.com/9/9/0x0/other-side.jpg'">
        </oem-gallery-component>
      </aside>
    </body></html>
    """
    detail = DetailSpec(
        root_selector="body",
        gallery_selector="oem-gallery-component",
        gallery_item_selector=None,
        fields={},
    )

    result = extract_vdp(
        html,
        detail_url=requested,
        origin="https://dealer.example",
        detail=detail,
        expected_vin=VIN,
    )

    assert result.identity_proven
    assert result.record["vin"] == VIN
    assert result.record["photos"] == [
        "https://content.homenetiol.com/2000157/2065512/0x0/front.jpg",
        "https://content.homenetiol.com/2000157/2065512/0x0/side.jpg",
        "https://content.homenetiol.com/2000157/2065512/0x0/rear.jpg",
    ]
    assert all(photo.source == "known_cdn_full" for photo in result.photos)
    assert all(photo.full_resolution_candidate for photo in result.photos)
    assert all("/9/9/" not in photo.url for photo in result.photos)


def test_vdp_accepts_gallery_when_every_selected_asset_names_expected_vin() -> None:
    html = f"""
    <html><head><link rel="canonical" href="{URL}"></head><body>
      <main class="vehicle-detail">
        <div class="car-detail-images__main">
          <a href="/media/front.jpg"><img src="/media/front.jpg" alt="Front photo {VIN}"></a>
          <a href="/media/side.jpg"><img src="/media/side.jpg" alt="Side photo {VIN}"></a>
          <a href="/media/rear.jpg"><img src="/media/rear.jpg" alt="Rear photo {VIN}"></a>
        </div>
      </main>
      <aside class="related-vehicles">
        <img src="/media/other.jpg" alt="1M8GDM9AXKP042788">
      </aside>
    </body></html>
    """
    detail = DetailSpec(
        root_selector="main.vehicle-detail",
        gallery_selector=".car-detail-images__main",
        gallery_item_selector="img",
        fields={},
    )

    result = extract_vdp(
        html,
        detail_url=URL,
        origin="https://dealer.example",
        detail=detail,
        expected_vin=VIN,
    )

    assert result.identity_proven
    assert result.record["photos"] == [
        "https://dealer.example/media/front.jpg",
        "https://dealer.example/media/side.jpg",
        "https://dealer.example/media/rear.jpg",
    ]
    assert all("other" not in photo.url for photo in result.photos)


def test_vdp_rejects_configured_gallery_if_one_asset_lacks_expected_vin() -> None:
    html = f"""
    <html><head><link rel="canonical" href="{URL}"></head><body>
      <main class="vehicle-detail">
        <div class="car-detail-images__main">
          <img src="/media/front.jpg" alt="Front photo {VIN}">
          <img src="/media/unlabelled.jpg" alt="Side photo">
        </div>
      </main>
    </body></html>
    """
    detail = DetailSpec(
        root_selector="main.vehicle-detail",
        gallery_selector=".car-detail-images__main",
        gallery_item_selector="img",
        fields={},
    )

    result = extract_vdp(
        html,
        detail_url=URL,
        origin="https://dealer.example",
        detail=detail,
        expected_vin=VIN,
    )

    assert result.identity_proven
    assert result.photos == ()
    assert "photos" not in result.record


def test_vin_bound_oem_gallery_attribute_is_owned_full_image_gallery() -> None:
    html = f"""
    <html><head><link rel="canonical" href="{URL}"></head><body>
      <main class="vehicle">
        <h1>Honda Civic</h1>
        <oem-gallery-component :vin="'{VIN}'"
          :photoUrls="'https://cdn.example/{VIN}-1.jpg,https://cdn.example/{VIN}-2.jpg'"></oem-gallery-component>
      </main>
    </body></html>
    """
    result = extract_vdp(
        html,
        detail_url=URL,
        origin="https://dealer.example",
        detail=DetailSpec(
            root_selector="main.vehicle",
            gallery_selector="oem-gallery-component",
            gallery_item_selector="img",
            fields={},
        ),
        expected_vin=VIN,
    )
    assert result.identity_proven
    assert result.record["vin"] == VIN
    assert len(result.photos) == 2
    assert all(photo.source in {"data_full", "vin_gallery_list", "known_cdn_full"} for photo in result.photos)


def test_vdp_rejects_single_custom_gallery_owned_by_a_different_vin() -> None:
    other_vin = "1M8GDM9AXKP042788"
    html = f"""
    <html><head><link rel="canonical" href="{URL}"></head><body>
      <main class="vehicle" data-vin="{VIN}">
        <oem-gallery-component
          :vin="'{other_vin}'"
          :photoUrls="'https://cdn.example/other-front.jpg,https://cdn.example/other-side.jpg'">
        </oem-gallery-component>
      </main>
    </body></html>
    """

    result = extract_vdp(
        html,
        detail_url=URL,
        origin="https://dealer.example",
        detail=DetailSpec(
            root_selector="main.vehicle",
            gallery_selector="oem-gallery-component",
            gallery_item_selector=None,
            fields={},
        ),
        expected_vin=VIN,
    )

    assert result.identity_proven
    assert result.record["vin"] == VIN
    assert result.photos == ()
    assert "photo" not in result.record
    assert "photos" not in result.record


def test_configured_vdp_root_can_use_exact_url_vin_for_dom_gallery() -> None:
    html = f"""
    <html><head><link rel="canonical" href="{URL}"></head><body>
      <main class="vehicle">
        <h1>Honda Civic {VIN}</h1>
        <section class="gallery">
          <a href="/photos/{VIN}-1.jpg"><img src="/thumbs/{VIN}-1.jpg" alt="Front {VIN}"></a>
          <a href="/photos/{VIN}-2.jpg"><img src="/thumbs/{VIN}-2.jpg" alt="Side {VIN}"></a>
        </section>
      </main>
    </body></html>
    """
    result = extract_vdp(
        html,
        detail_url=URL,
        origin="https://dealer.example",
        detail=DetailSpec(
            root_selector="main.vehicle",
            gallery_selector=".gallery",
            gallery_item_selector="img",
            fields={},
        ),
        expected_vin=VIN,
    )
    assert result.identity_proven
    assert len(result.photos) == 2


def _next_flight_script(record_id: str, value: object) -> str:
    payload = f"{record_id}:{json.dumps(value, separators=(',', ':'))}"
    return f"self.__next_f.push({json.dumps([1, payload])})"


def test_vdp_extracts_one_rich_vin_bound_next_flight_vehicle() -> None:
    centre_vin = "JN8BT3BB7RW435564"
    centre_url = (
        "https://www.centreautomobilesduquebec.com/fr/vehicle/"
        "nissan-rogue-2024-6a4d40fa21a956acd85504af"
    )
    originals = [
        "https://megavehicules.com/uplfoto/uploads/DA245063/M-3434/11781781729179.jpg",
        "https://megavehicules.com/uplfoto/uploads/DA245063/M-3434/15391781729180.jpg",
        "https://megavehicules.com/uplfoto/uploads/DA245063/M-3434/16201781729182.jpg",
    ]
    primary = {
        "serialnumber": centre_vin,
        "heading": "Nissan Rogue 2024",
        "make": "Nissan",
        "model": "Rogue",
        "year": 2024,
        "trim": "SV Premium AWD",
        "stocknumber": "M-3434",
        "price": 30990,
        "km": 12597,
        "base_ext_color": "bleu",
        "base_int_color": "noir",
        "transtype": "Automatique",
        "description_fr": "Rogue AWD avec sièges chauffants.",
        "photo": originals,
    }
    # A normal related card is present in the same Flight stream, but owns one
    # thumbnail rather than a complete page gallery.
    related = {
        "vin": "1M8GDM9AXKP042788",
        "year": 2024,
        "make": "Honda",
        "model": "Pilot",
        "price": 40000,
        "mileage": 20,
        "images": ["https://megavehicules.com/uplfoto/uploads/X/Y/related.jpg"],
    }
    html = f"""
    <html><head><link rel="canonical" href="{centre_url}"></head><body>
      <script>{_next_flight_script('6', ['$', '$L18', None, {'car': primary}])}</script>
      <script>{_next_flight_script('7', ['$', '$L19', None, {'car': related}])}</script>
    </body></html>
    """

    result = extract_vdp(
        html,
        detail_url=centre_url,
        origin="https://www.centreautomobilesduquebec.com",
        detail=DetailSpec(root_selector=None, fields={}),
        expected_vin=None,
    )

    assert result.identity_proven
    assert result.matched_by == "next_flight:unambiguous_vin"
    assert result.record["vin"] == centre_vin
    assert result.record["price"] == 30990
    assert result.record["mileage"] == 12597
    assert result.record["distance_unit"] == "km"
    assert result.record["color_ext"] == "bleu"
    assert result.record["color_int"] == "noir"
    assert result.record["transmission"] == "Automatique"
    assert result.record["description"] == "Rogue AWD avec sièges chauffants."
    assert result.record["photos"] == originals
    assert all(photo.source == "known_cdn_full" for photo in result.photos)
    assert all("related" not in photo.url for photo in result.photos)


def test_vdp_extracts_unique_ridemotive_vehicle_from_batched_flight_rows() -> None:
    wyler_vin = "KNAFU4A23A5310456"
    wyler_url = (
        "https://wylereastgate.com/inventory/"
        "Used-2010-Kia-Forte-EX-KNAFU4A23A5310456-2897"
    )
    image_ids = [
        "3uagz5xuc6ikvos5ax35zxijm08g",
        "q1ctz4ybj74oiguk806pbslku93j",
        "2bm7711s4zhgjlggh5np1jgbn7vu",
    ]
    webp_ids = [
        "ssqbtikl16aw7nsbvk31tnzdxz8l",
        "z5i72wny085b4chlikt4hfc5ml37",
        "3gl81j4ck06nozz43ie2pg4t7j54",
    ]
    vehicle = {
        "vin": wyler_vin,
        "make_year": 2010,
        "make": "Kia",
        "model": "Forte",
        "stock_number": "11A5310456T1",
        "price": 4722,
        "odometer": 118799,
        "exterior_color": "Ebony Black",
        "transmission": "Automatic",
        "description": "$a3",
        "images": image_ids,
        "webp_images": webp_ids,
    }
    record = [["$", "$L1", None, {"vehicle": vehicle}]]
    # Flight text rows are length-delimited in bytes and can be followed by a
    # JSON row without a newline. A raw substring search would be unsafe here.
    text_value = "Résumé 🚗"
    text_bytes = text_value.encode("utf-8")
    decoy = [{"routerNoise": index} for index in range(2_100)]
    payload = (
        "1:I[123,[],\"default\"]\n:HL[\"/app.css\",\"style\"]\n"
        f"0:{json.dumps(decoy, separators=(',', ':'))}\n"
        f"a3:T{len(text_bytes):x},{text_value}"
        f"7:{json.dumps(record, separators=(',', ':'))}"
    )
    script = f"self.__next_f.push({json.dumps([1, payload])})"
    schema_description = "A practical Kia Forte with a moonroof and efficient four-cylinder engine."
    schema = {
        "@context": "https://schema.org",
        "@type": "Vehicle",
        "vehicleIdentificationNumber": wyler_vin,
        "description": schema_description,
        "url": wyler_url,
    }
    html = f"""
    <html><head><link rel="canonical" href="{wyler_url}">
      <script type="application/ld+json">{json.dumps(schema)}</script>
    </head><body>
      <script>{script}</script>
      <img src="https://images.app.ridemotive.com/4qm8ffoawc3rntq72x9l62s8a2b5"
           alt="Dealer logo">
    </body></html>
    """

    result = extract_vdp(
        html,
        detail_url=wyler_url,
        origin="https://wylereastgate.com",
        detail=DetailSpec(root_selector=None, fields={}),
        expected_vin=wyler_vin,
    )

    expected = [
        f"https://images.app.ridemotive.com/{image_id}"
        for image_id in image_ids
    ]
    assert result.identity_proven
    assert result.matched_by == "next_flight:vin"
    assert result.record["vin"] == wyler_vin
    assert result.record["price"] == 4722
    assert result.record["mileage"] == 118799
    assert result.record["description"] == schema_description
    assert result.record["photos"] == expected
    assert all(photo.source == "known_cdn_full" for photo in result.photos)
    assert all(photo.width == 1024 for photo in result.photos)
    assert all("4qm8ff" not in photo.url for photo in result.photos)


def test_vdp_rejects_ambiguous_ridemotive_vehicle_mappings_and_raw_js() -> None:
    first_ids = [
        "3uagz5xuc6ikvos5ax35zxijm08g",
        "q1ctz4ybj74oiguk806pbslku93j",
    ]
    second_ids = [
        "2bm7711s4zhgjlggh5np1jgbn7vu",
        "ku19tw2q8orxu1nk7zc9bzqerz4h",
    ]

    def vehicle(vin: str, images: list[str]) -> dict[str, object]:
        return {
            "vin": vin,
            "year": 2020,
            "make": "Kia",
            "model": "Forte",
            "price": 20000,
            "mileage": 100,
            "stock": vin[-6:],
            "images": images,
            "webp_images": list(reversed(images)),
        }

    rows = [
        {"vehicle": vehicle(VIN, first_ids)},
        {"vehicle": vehicle("1M8GDM9AXKP042788", second_ids)},
    ]
    payload = f"7:{json.dumps(rows, separators=(',', ':'))}"
    script = f"self.__next_f.push({json.dumps([1, payload])})"
    raw_decoy = json.dumps({"vehicle": vehicle(VIN, first_ids)})
    html = f"""
    <html><head><link rel="canonical" href="{URL}"></head><body>
      <script>{script}</script>
      <script>window.__unsafe_decoy = {raw_decoy};</script>
    </body></html>
    """

    result = extract_vdp(
        html,
        detail_url=URL,
        origin="https://dealer.example",
        detail=DetailSpec(root_selector=None, fields={}),
        expected_vin=VIN,
    )

    assert result.identity_proven
    assert result.photos == ()
    assert "photos" not in result.record


def test_vdp_does_not_scan_past_malformed_flight_text_length() -> None:
    vehicle = {
        "vin": VIN,
        "year": 2025,
        "make": "Honda",
        "model": "Civic",
        "price": 25000,
        "mileage": 10,
        "stock": "ABC123",
        "images": [
            "3uagz5xuc6ikvos5ax35zxijm08g",
            "q1ctz4ybj74oiguk806pbslku93j",
        ],
        "webp_images": [
            "2bm7711s4zhgjlggh5np1jgbn7vu",
            "ku19tw2q8orxu1nk7zc9bzqerz4h",
        ],
    }
    fake_row = f"7:{json.dumps({'vehicle': vehicle}, separators=(',', ':'))}"
    payload = f"a3:Tfffffff,short{fake_row}"
    script = f"self.__next_f.push({json.dumps([1, payload])})"
    html = f"""
    <html><head><link rel="canonical" href="{URL}"></head><body>
      <script>{script}</script>
    </body></html>
    """

    result = extract_vdp(
        html,
        detail_url=URL,
        origin="https://dealer.example",
        detail=DetailSpec(root_selector=None, fields={}),
        expected_vin=VIN,
    )

    assert result.photos == ()


def test_vdp_reads_consensus_price_from_direct_schema_offer_list() -> None:
    schema = {
        "@context": "https://schema.org",
        "@type": "Vehicle",
        "vehicleIdentificationNumber": VIN,
        "vehicleModelDate": 2012,
        "brand": "Nissan",
        "model": "Altima",
        "offers": [
            {"@type": "Offer", "price": "$5,815"},
            {"@type": "Offer", "price": 5815},
        ],
        "image": "https://cdn.example/altima.jpg",
        "url": URL,
    }
    html = f"""
    <html><head><link rel="canonical" href="{URL}">
      <script type="application/ld+json">{json.dumps(schema)}</script>
    </head><body></body></html>
    """

    result = extract_vdp(
        html,
        detail_url=URL,
        origin="https://dealer.example",
        detail=DetailSpec(root_selector=None, fields={}),
        expected_vin=VIN,
    )

    assert result.record["price"] == 5815


def test_vdp_rejects_conflicting_or_nested_offer_list_prices() -> None:
    schema = {
        "@context": "https://schema.org",
        "@type": "Vehicle",
        "vehicleIdentificationNumber": VIN,
        "vehicleModelDate": 2012,
        "brand": "Nissan",
        "model": "Altima",
        "offers": [
            {"@type": "Offer", "price": 5815},
            {
                "@type": "Offer",
                "price": 5999,
                "seller": {"price": 1},
            },
            "not-an-offer",
        ],
        "url": URL,
    }
    html = f"""
    <html><head><link rel="canonical" href="{URL}">
      <script type="application/ld+json">{json.dumps(schema)}</script>
    </head><body></body></html>
    """

    result = extract_vdp(
        html,
        detail_url=URL,
        origin="https://dealer.example",
        detail=DetailSpec(root_selector=None, fields={}),
        expected_vin=VIN,
    )

    assert "price" not in result.record


def test_vdp_reads_unique_identity_scoped_semantic_description() -> None:
    description = (
        "This Ford Escape combines an efficient hybrid drivetrain with all-wheel "
        "drive, heated seating, adaptive cruise control, and a panoramic roof."
    )
    html = f"""
    <html><head><link rel="canonical" href="{URL}"></head><body>
      <article class="vehicle-detail" data-vin="{VIN}">
        <section id="tab-description">{description}</section>
        <aside class="related-vehicles">
          <div class="vehicle-description">Unrelated vehicle copy that must not win.</div>
        </aside>
      </article>
    </body></html>
    """

    result = extract_vdp(
        html,
        detail_url=URL,
        origin="https://dealer.example",
        detail=DetailSpec(root_selector="article.vehicle-detail", fields={}),
        expected_vin=VIN,
    )

    assert result.identity_proven
    assert result.record["description"] == description


def test_vdp_decodes_only_direct_flat_json_image_list_from_flight_state() -> None:
    beaucage_vin = "1FMCU9H6XLUA13474"
    beaucage_url = (
        "https://www.occasionbeaucage.com/auto-usage/"
        "ford-escape-2020-midtm030a/"
    )
    originals = [
        "https://evalauto-resources.s3.us-east-2.amazonaws.com/concession/QC/24/cars/305866/1a034429a0b6af0cf44.66322470.jpg",
        "https://evalauto-resources.s3.us-east-2.amazonaws.com/concession/QC/24/cars/305866/1a03442a2122550ddc3.20062659.jpg",
        "https://evalauto-resources.s3.us-east-2.amazonaws.com/concession/QC/24/cars/305866/1a03442aba4249d0c16.90132377.jpg",
    ]
    vehicle = {
        "vin": beaucage_vin,
        "year": 2020,
        "make": "Ford",
        "model": "Escape",
        "price": "16\u00a0995\u00a0$",
        "mileage_from_odometer": "99\u00a0000",
        "mileage_from_odometer_unit_code": "km",
        "exterior_color": "Noir",
        "transmission": "Automatique",
        "short_description_localized": "Automatique, moteur 1.5L.",
        "images": json.dumps(originals),
    }
    html = f"""
    <html><head><link rel="canonical" href="{beaucage_url}"></head><body>
      <script>{_next_flight_script('1c', ['$', '$L1f', None, {'$contextData': {'vehicle': vehicle}}])}</script>
    </body></html>
    """

    result = extract_vdp(
        html,
        detail_url=beaucage_url,
        origin="https://www.occasionbeaucage.com",
        detail=DetailSpec(root_selector=None, fields={}),
        expected_vin=None,
    )

    assert result.identity_proven
    assert result.record["vin"] == beaucage_vin
    assert result.record["price"] == 16995
    assert result.record["mileage"] == 99000
    assert result.record["distance_unit"] == "km"
    assert result.record["photos"] == originals
    assert result.record["description"] == "Automatique, moteur 1.5L."
    assert all(photo.source == "known_cdn_full" for photo in result.photos)


def test_vdp_rejects_ambiguous_multi_vehicle_flight_galleries() -> None:
    first = {
        "vin": VIN,
        "year": 2025,
        "make": "Honda",
        "model": "Civic",
        "price": 25000,
        "mileage": 10,
        "images": ["https://cdn.example/a1.jpg", "https://cdn.example/a2.jpg"],
    }
    other_vin = "1M8GDM9AXKP042788"
    second = {
        "vin": other_vin,
        "year": 2024,
        "make": "Honda",
        "model": "Pilot",
        "price": 40000,
        "mileage": 20,
        "images": ["https://cdn.example/b1.jpg", "https://cdn.example/b2.jpg"],
    }
    html = f"""
    <html><head><link rel="canonical" href="{URL}"></head><body>
      <input name="vin" value="{VIN}">
      <script>{_next_flight_script('6', ['$', '$L18', None, {'car': first}])}</script>
      <script>{_next_flight_script('7', ['$', '$L19', None, {'car': second}])}</script>
    </body></html>
    """

    result = extract_vdp(
        html,
        detail_url=URL,
        origin="https://dealer.example",
        detail=DetailSpec(root_selector=None, fields={}),
        expected_vin=VIN,
    )

    assert result.identity_proven
    assert result.photos == ()
    assert "photos" not in result.record


def test_vdp_accepts_exact_same_slug_canonical_alias_and_wordpress_originals() -> None:
    ridetime_vin = "1FMCU9DZ1MUA74472"
    slug = "26202-ford-escape-titanium-hybrid-2021-winnipeg-mb"
    requested = f"https://www.ridetime.ca/used-cars/{slug}/"
    canonical = f"https://www.ridetime.ca/car-parts/{slug}/"
    first = (
        "https://www.ridetime.ca/wp-content/uploads/2026/08/"
        "automobiles-used-2021-ford-escape-1028021-primary-photo-Image.jpg"
    )
    second = (
        "https://www.ridetime.ca/wp-content/uploads/2026/08/"
        "automobiles-used-2021-ford-escape-1770882-left-front-photo-Image.jpg"
    )
    html = f"""
    <html><head><link rel="canonical" href="{canonical}"></head><body>
      <main class="vehicle"><div class="carproof-badge" data-vin="{ridetime_vin}"></div>
        <div class="carousel--product-gallery">
          <img src="{first}" alt="Ford Escape Titanium Hybrid 2021">
          <img src="{second}" alt="Ford Escape Titanium Hybrid 2021">
        </div>
      </main>
      <aside class="related-vehicles">
        <img src="https://www.ridetime.ca/wp-content/uploads/2026/08/related.jpg">
      </aside>
    </body></html>
    """

    result = extract_vdp(
        html,
        detail_url=requested,
        origin="https://www.ridetime.ca",
        detail=DetailSpec(
            root_selector="main.vehicle",
            gallery_selector=".carousel--product-gallery",
            gallery_item_selector="img",
            fields={},
        ),
        expected_vin=None,
    )

    assert result.identity_proven
    assert result.record["vin"] == ridetime_vin
    assert result.record["photos"] == [first, second]
    assert all(photo.source == "known_cdn_full" for photo in result.photos)
    assert all("related" not in photo.url for photo in result.photos)


def test_vdp_rejects_canonical_alias_with_different_inventory_slug() -> None:
    ridetime_vin = "1FMCU9DZ1MUA74472"
    requested = "https://www.ridetime.ca/used-cars/26202-ford-escape-2021/"
    canonical = "https://www.ridetime.ca/car-parts/99999-ford-escape-2021/"
    html = f"""
    <html><head><link rel="canonical" href="{canonical}"></head><body>
      <main class="vehicle"><div data-vin="{ridetime_vin}"></div>
        <div class="carousel--product-gallery">
          <img src="https://www.ridetime.ca/wp-content/uploads/2026/08/a.jpg">
          <img src="https://www.ridetime.ca/wp-content/uploads/2026/08/b.jpg">
        </div>
      </main>
    </body></html>
    """

    result = extract_vdp(
        html,
        detail_url=requested,
        origin="https://www.ridetime.ca",
        detail=DetailSpec(
            root_selector="main.vehicle",
            gallery_selector=".carousel--product-gallery",
            gallery_item_selector="img",
            fields={},
        ),
        expected_vin=None,
    )

    assert not result.identity_proven
    assert result.photos == ()


def test_vdp_records_exact_dealer_com_1024_width_gallery_evidence() -> None:
    huber_vin = "WAUAUDGY7RA039670"
    huber_url = (
        "https://www.huberautomotive.com/used/Audi/2024-Audi-A3-"
        "b49b49ceac1840287e9654751cd57871.htm"
    )
    first = (
        "https://pictures.dealer.com/h/huberauto/1462/"
        "122110bcada166b51b86c9b4e7c4dd2bx.jpg?"
        "impolicy=downsize_bkpt&imdensity=1&w=1024"
    )
    second = (
        "https://pictures.dealer.com/h/huberauto/1913/"
        "a22ad14643dd8ebca2d2065d2b60369bx.jpg?"
        "impolicy=downsize_bkpt&imdensity=1&w=1024"
    )
    schema = {
        "@context": "https://schema.org",
        "@type": ["Product", "Car"],
        "vehicleIdentificationNumber": huber_vin,
        "vehicleModelDate": 2024,
        "brand": {"@type": "Brand", "name": "Audi"},
        "model": "A3",
        "offers": {"@type": "Offer", "price": 20276},
        "image": first,
        "url": huber_url,
    }
    html = f"""
    <html><head><link rel="canonical" href="{huber_url}">
      <script type="application/ld+json">{json.dumps(schema)}</script>
    </head><body><input name="vin" value="{huber_vin}">
      <main class="vehicle"><div id="vehicle-gallery1-app-root">
        <img src="{first}"><img src="{second}">
      </div></main>
    </body></html>
    """

    result = extract_vdp(
        html,
        detail_url=huber_url,
        origin="https://www.huberautomotive.com",
        detail=DetailSpec(
            root_selector="main.vehicle",
            gallery_selector="#vehicle-gallery1-app-root",
            gallery_item_selector="img",
            fields={},
        ),
        expected_vin=None,
    )

    assert result.identity_proven
    assert result.record["photos"] == [first, second]
    assert all(photo.source == "known_cdn_full" for photo in result.photos)
    assert all(photo.width == 1024 for photo in result.photos)
    assert all(photo.full_resolution_candidate for photo in result.photos)


def test_vdp_uses_exact_vin_bound_dealer_com_gallery_state_not_thumbnails() -> None:
    huber_vin = "WAUAUDGY7RA039670"
    huber_url = "https://www.huberautomotive.com/used/Audi/a3-id.htm"
    originals = [
        "https://pictures.dealer.com/h/huberauto/1462/122110bcada166b51b86c9b4e7c4dd2bx.jpg",
        "https://pictures.dealer.com/h/huberauto/1913/a22ad14643dd8ebca2d2065d2b60369bx.jpg",
    ]
    state = {
        "media": {
            "images": [
                {
                    "src": original,
                    "thumbnail": original.rsplit("/", 1)[0]
                    + "/thumb_"
                    + original.rsplit("/", 1)[1],
                }
                for original in originals
            ]
        },
        "requestData": {"vin": huber_vin},
    }
    html = f"""
    <html><head><link rel="canonical" href="{huber_url}"></head><body>
      <script>DDC.WS.state['ws-vehicle-gallery']['vehicle-gallery1'] = {json.dumps(state)};</script>
    </body></html>
    """

    result = extract_vdp(
        html,
        detail_url=huber_url,
        origin="https://www.huberautomotive.com",
        detail=DetailSpec(root_selector=None, fields={}),
        expected_vin=None,
    )

    assert result.identity_proven
    assert result.record["vin"] == huber_vin
    assert result.record["photos"] == originals
    assert all("thumb_" not in photo.url for photo in result.photos)
    assert all(photo.source == "known_cdn_full" for photo in result.photos)


def test_vdp_rejects_dealer_com_gallery_state_with_two_vins() -> None:
    state = {
        "media": {
            "images": [
                {"src": "https://pictures.dealer.com/h/dealer/1/a.jpg"},
                {"src": "https://pictures.dealer.com/h/dealer/2/b.jpg"},
            ]
        },
        "requestData": {"vin": VIN},
        "relatedVehicle": {"vin": "1M8GDM9AXKP042788"},
    }
    html = f"""
    <html><head><link rel="canonical" href="{URL}"></head><body>
      <input name="vin" value="{VIN}">
      <script>DDC.WS.state['ws-vehicle-gallery']['vehicle-gallery1'] = {json.dumps(state)};</script>
    </body></html>
    """

    result = extract_vdp(
        html,
        detail_url=URL,
        origin="https://dealer.example",
        detail=DetailSpec(root_selector=None, fields={}),
        expected_vin=VIN,
    )

    assert result.identity_proven
    assert result.photos == ()


def test_dealereprocess_vdp_gallery_normalizes_only_one_immutable_vehicle_album() -> None:
    vin = "1N4AL2EP5CC227965"
    detail_url = "https://dealer.example/auto/used-2012-nissan-altima-25-s/123493676/"
    schema = {
        "@context": "https://schema.org",
        "@type": "Vehicle",
        "vehicleIdentificationNumber": vin,
        "vehicleModelDate": 2012,
        "brand": "Nissan",
        "model": "Altima",
        "url": "/auto/used-2012-nissan-altima-25-s/123493676/",
    }
    related_schema = {
        "@context": "https://schema.org",
        "@type": "Vehicle",
        "vehicleIdentificationNumber": "3N1AB7AP6DL686723",
        "vehicleModelDate": 2013,
        "brand": "Nissan",
        "model": "Sentra",
        "url": "/auto/used-2013-nissan-sentra-sv/123939758/",
    }
    prefix = "https://cloudflareimages.dealereprocess.com/resrc/images/c_limit,fl_lossy,w_900/v1/dvp/4170"
    first = f"{prefix}/53188861974/Used-2012-Nissan-Altima-ID53188861974-front=="
    second = f"{prefix}/53188861989/Used-2012-Nissan-Altima-ID53188861989-side=="
    html = f"""
    <html><head><link rel="canonical" href="{detail_url}">
      <script type="application/ld+json">{json.dumps(schema)}</script>
      <script type="application/ld+json">{json.dumps(related_schema)}</script>
    </head><body><main class="vehicle-detail">
      <div class="vehicle_loopslider vehicle_loopslider--vdp">
        <img src="{first}" alt="Used 2012 Nissan Altima 2.5 S">
        <img src="{second}" alt="Used 2012 Nissan Altima 2.5 S">
      </div>
    </main></body></html>
    """

    result = extract_vdp(
        html,
        detail_url=detail_url,
        origin="https://dealer.example",
        detail=DetailSpec(
            root_selector="main.vehicle-detail",
            gallery_selector=".vehicle_loopslider--vdp",
            gallery_item_selector="img",
            fields={},
        ),
        expected_vin=None,
    )

    assert result.identity_proven
    assert result.matched_by == "json_ld:url"
    assert result.record["year"] == 2012
    assert result.record["make"] == "Nissan"
    assert result.record["model"] == "Altima"
    assert result.record["photos"] == [
        first.replace("w_900", "w_1920"),
        second.replace("w_900", "w_1920"),
    ]
    assert all(photo.width == 1920 for photo in result.photos)
    assert all(photo.source == "known_cdn_full" for photo in result.photos)


def test_dealereprocess_gallery_rejects_a_second_inventory_album() -> None:
    vin = "1N4AL2EP5CC227965"
    detail_url = "https://dealer.example/auto/used-altima/123493676/"
    schema = {
        "@context": "https://schema.org",
        "@type": "Vehicle",
        "vehicleIdentificationNumber": vin,
        "url": detail_url,
    }
    base = "https://cloudflareimages.dealereprocess.com/resrc/images/fl_lossy,w_900/v1/dvp/4170"
    html = f"""
    <html><head><link rel="canonical" href="{detail_url}">
      <script type="application/ld+json">{json.dumps(schema)}</script>
    </head><body><main class="vehicle-detail">
      <div class="vehicle_loopslider vehicle_loopslider--vdp">
        <img src="{base}/firstAsset123/Used-Nissan-ID123493676-front">
        <img src="{base}/otherAsset123/Used-Toyota-ID999999999-side">
      </div>
    </main></body></html>
    """

    result = extract_vdp(
        html,
        detail_url=detail_url,
        origin="https://dealer.example",
        detail=DetailSpec(
            root_selector="main.vehicle-detail",
            gallery_selector=".vehicle_loopslider--vdp",
            gallery_item_selector="img",
            fields={},
        ),
        expected_vin=None,
    )

    assert result.identity_proven
    assert result.photos == ()


def test_remora_gallery_requires_the_expected_vin_in_every_photo_url() -> None:
    vin = "2LMDJ8JK1BBJ12188"
    detail_url = f"https://dealer.example/for-sale/used-lincoln-mkx/{vin}"
    first = f"https://vimg.remora.inc/547371/{vin.lower()}-front-1-113219.avif"
    second = f"https://vimg.remora.inc/547371/{vin.lower()}-side-2-113219.avif"
    html = f"""
    <html><head><link rel="canonical" href="{detail_url}"></head><body>
      <main class="vehicle-detail"><div class="gallery ui wide">
        <picture><source srcset="{first.replace('.avif', '.thumb.avif')}"><img src="{first}"></picture>
        <picture><source srcset="{second.replace('.avif', '.thumb.avif')}"><img src="{second}"></picture>
      </div></main>
    </body></html>
    """

    result = extract_vdp(
        html,
        detail_url=detail_url,
        origin="https://dealer.example",
        detail=DetailSpec(
            root_selector="main.vehicle-detail",
            gallery_selector=".gallery.ui.wide",
            gallery_item_selector="img",
            fields={},
        ),
        expected_vin=None,
    )

    assert result.identity_proven
    assert result.record["photos"] == [first, second]
    assert all(".thumb." not in photo.url for photo in result.photos)
    assert all(photo.source == "known_cdn_full" for photo in result.photos)


def test_autoscout_gallery_strips_only_renditions_from_structured_primary_album() -> None:
    vin = "2FMHK6DC6CBD14108"
    detail_url = "https://dealer.example/vehicles/2012/ford/flex/70619517/"
    album = "07c309fe-179e-4bd4-a887-8f1eac25f6d4"
    first = (
        "https://prod.pictures.autoscout24.net/listing-images/"
        f"{album}_97b43910-c064-4c3f-b03b-83faed41f45d.jpg"
    )
    second = (
        "https://prod.pictures.autoscout24.net/listing-images/"
        f"{album}_10b43910-c064-4c3f-b03b-83faed41f45d.jpg"
    )
    schema = {
        "@context": "https://schema.org",
        "@type": "Vehicle",
        "vehicleIdentificationNumber": vin,
        "url": detail_url,
        "image": first,
    }
    html = f"""
    <html><head><link rel="canonical" href="{detail_url}">
      <script type="application/ld+json">{json.dumps(schema)}</script>
    </head><body><main class="vehicle-detail"><div class="photo-gallery">
      <img src="{first}/1024x786.webp">
      <img src="{first}/133x100.webp">
      <img src="{second}/133x100.webp">
      <aside class="related-vehicles">
        <img src="https://prod.pictures.autoscout24.net/listing-images/aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa_other-car.jpg/133x100.webp">
      </aside>
    </div></main></body></html>
    """

    result = extract_vdp(
        html,
        detail_url=detail_url,
        origin="https://dealer.example",
        detail=DetailSpec(
            root_selector="main.vehicle-detail",
            gallery_selector=".photo-gallery",
            gallery_item_selector="img",
            fields={},
        ),
        expected_vin=None,
    )

    assert result.identity_proven
    assert result.record["photos"] == [first, second]
    assert len({photo.url for photo in result.photos}) == 2
    assert all(
        "1024x786" not in photo.url and "133x100" not in photo.url
        for photo in result.photos
    )
    assert all("other-car" not in photo.url for photo in result.photos)
    assert all(photo.source == "known_cdn_full" for photo in result.photos)
    assert all(photo.width is None for photo in result.photos)
    assert all(photo.full_resolution_candidate for photo in result.photos)


def test_autoscout_original_does_not_inherit_thumbnail_layout_width() -> None:
    vin = "2FMHK6DC6CBD14108"
    detail_url = "https://dealer.example/vehicles/2012/ford/flex/70619517/"
    album = "07c309fe-179e-4bd4-a887-8f1eac25f6d4"
    first = (
        "https://prod.pictures.autoscout24.net/listing-images/"
        f"{album}_97b43910-c064-4c3f-b03b-83faed41f45d.jpg"
    )
    second = (
        "https://prod.pictures.autoscout24.net/listing-images/"
        f"{album}_10b43910-c064-4c3f-b03b-83faed41f45d.jpg"
    )
    schema = {
        "@context": "https://schema.org",
        "@type": "Vehicle",
        "vehicleIdentificationNumber": vin,
        "url": detail_url,
        "image": first,
    }
    html = f"""
    <html><head><link rel="canonical" href="{detail_url}">
      <script type="application/ld+json">{json.dumps(schema)}</script>
    </head><body><main class="vehicle-detail"><div class="photo-gallery">
      <img src="{first}" width="133"><img src="{second}" width="133">
    </div></main></body></html>
    """

    result = extract_vdp(
        html,
        detail_url=detail_url,
        origin="https://dealer.example",
        detail=DetailSpec(
            root_selector="main.vehicle-detail",
            gallery_selector=".photo-gallery",
            gallery_item_selector="img",
            fields={},
        ),
        expected_vin=vin,
    )

    assert result.record["photos"] == [first, second]
    assert all(photo.source == "known_cdn_full" for photo in result.photos)
    assert all(photo.width is None for photo in result.photos)
    assert all(photo.full_resolution_candidate for photo in result.photos)


def test_sm360_gallery_removes_renditions_and_rejects_overlay_assets() -> None:
    vin = "KNDEUCAA3P7374977"
    detail_url = "https://dealer.example/en/used-inventory/kia/seltos/2023-kia-seltos-id38356554"
    root = "https://img.sm360.ca/images/inventory/401dixiekia-376/kia/seltos/2023/38356554"
    first = f"{root}/asset-front-123.jpeg"
    second = f"{root}/asset-side-456.jpeg"
    overlay = (
        "https://img.sm360.ca/images/web/dilawri-group-of-companies/3792/"
        "dilawri_overlay_v2.png"
    )
    schema = {
        "@context": "https://schema.org",
        "@type": "Vehicle",
        "vehicleIdentificationNumber": vin,
        "url": detail_url,
        "image": first,
    }
    html = f"""
    <html><head><link rel="canonical" href="{detail_url}">
      <script type="application/ld+json">{json.dumps(schema)}</script>
    </head><body><main class="vehicle-detail">
      <div class="inventory-details-infos__gallery">
        <img src="{first.replace('/images/', '/ir/w640h480/images/')}">
        <img class="widget-ninjabox__watermark" src="{overlay}">
        <img src="{second.replace('/images/', '/ir/w400h300c/images/')}">
      </div>
    </main></body></html>
    """

    result = extract_vdp(
        html,
        detail_url=detail_url,
        origin="https://dealer.example",
        detail=DetailSpec(
            root_selector="main.vehicle-detail",
            gallery_selector=".inventory-details-infos__gallery",
            gallery_item_selector="img",
            fields={},
        ),
        expected_vin=None,
    )

    assert result.identity_proven
    assert result.record["photos"] == [first, second]
    assert all("/ir/" not in photo.url for photo in result.photos)
    assert all("overlay" not in photo.url for photo in result.photos)


def test_birchwood_vehicle_carousel_accepts_only_large_album_on_same_vdp_path() -> None:
    vin = "1C4RJKEG9P8123456"
    canonical = "https://www.birchwood.ca/vehicles/2023/jeep-grandcherokeel-summitreserve/F7JGMV/"
    requested = f"{canonical}?finance_type=finance"
    first = "https://vehicle-photos.birchwood.ca/photos/vehicles/326185/100001-large.jpg"
    second = "https://vehicle-photos.birchwood.ca/photos/vehicles/326185/100002-large.jpg"
    schema = {
        "@context": "https://schema.org",
        "@type": "Vehicle",
        "vehicleIdentificationNumber": vin,
        "url": canonical,
    }
    html = f"""
    <html><head><link rel="canonical" href="{canonical}">
      <script type="application/ld+json">{json.dumps(schema)}</script>
    </head><body><main class="vehicle-detail">
      <div class="phoenix4-vehiclecarouselheader-widget">
        <ul class="ks-carousel-slide-list vehicle-carousel">
          <li><img src="{first}"></li><li><img src="{second}"></li>
        </ul>
        <ul class="ks-carousel-slide-pickers">
          <li><img src="{first.replace('-large', '-small')}"></li>
        </ul>
      </div>
    </main></body></html>
    """

    result = extract_vdp(
        html,
        detail_url=requested,
        origin="https://www.birchwood.ca",
        detail=DetailSpec(
            root_selector="main.vehicle-detail",
            gallery_selector="ul.ks-carousel-slide-list",
            gallery_item_selector="img",
            fields={},
        ),
        expected_vin=None,
    )

    assert result.identity_proven
    assert result.record["photos"] == [first, second]
    assert all("-small.jpg" not in photo.url for photo in result.photos)
    assert all(photo.source == "known_cdn_full" for photo in result.photos)


def test_presentation_query_exception_never_ignores_vehicle_selecting_query() -> None:
    vin = "1C4RJKEG9P8123456"
    canonical = "https://www.birchwood.ca/vehicles/2023/jeep/F7JGMV/"
    requested = f"{canonical}?vehicle_id=DIFFERENT"
    schema = {
        "@context": "https://schema.org",
        "@type": "Vehicle",
        "vehicleIdentificationNumber": vin,
        "url": canonical,
    }
    html = f"""
    <html><head><link rel="canonical" href="{canonical}">
      <script type="application/ld+json">{json.dumps(schema)}</script>
    </head><body><main><ul class="vehicle-carousel">
      <li><img src="https://vehicle-photos.birchwood.ca/photos/vehicles/1/1-large.jpg"></li>
      <li><img src="https://vehicle-photos.birchwood.ca/photos/vehicles/1/2-large.jpg"></li>
    </ul></main></body></html>
    """

    result = extract_vdp(
        html,
        detail_url=requested,
        origin="https://www.birchwood.ca",
        detail=DetailSpec(
            root_selector="main",
            gallery_selector="ul.vehicle-carousel",
            gallery_item_selector="img",
            fields={},
        ),
        expected_vin=vin,
    )

    assert not result.identity_proven
    assert result.photos == ()


def test_identity_owned_exact_colour_spec_is_deterministic_and_conflict_closed() -> None:
    vin = "1FMCU9DZ1MUA74472"
    detail_url = f"https://dealer.example/used-cars/26202-ford-escape-{vin}/"
    base_html = f"""
    <html><head><link rel="canonical" href="{detail_url}"></head><body>
      <main class="vehicle-detail" data-vin="{vin}">
        {{specs}}
      </main>
    </body></html>
    """
    detail = DetailSpec(root_selector="main.vehicle-detail", fields={})
    one = extract_vdp(
        base_html.format(
            specs=(
                '<ul><li class="specs__item"><span class="specs__label">Colour</span>'
                '<strong class="specs__value">Blue</strong></li></ul>'
            )
        ),
        detail_url=detail_url,
        origin="https://dealer.example",
        detail=detail,
        expected_vin=vin,
    )
    conflict = extract_vdp(
        base_html.format(
            specs=(
                '<ul><li><span class="specs__label">Exterior Color</span>'
                '<strong class="specs__value">Blue</strong></li>'
                '<li><span class="specs__label">Colour</span>'
                '<strong class="specs__value">Red</strong></li></ul>'
            )
        ),
        detail_url=detail_url,
        origin="https://dealer.example",
        detail=detail,
        expected_vin=vin,
    )

    assert one.record["color_ext"] == "Blue"
    assert "color_ext" not in conflict.record


def test_social_preview_rejects_same_dom_asset_explicitly_labeled_dealer_logo() -> None:
    logo = (
        "https://bucket.dealervenom.com/2022/08/Jim-norton-toyota-min.png"
        "?auto=compress%2Cformat&ixlib=php-1.2.1"
    )
    html = f"""
    <html><head><link rel="canonical" href="{URL}">
      <meta property="og:image" content="{logo}">
    </head><body><main class="vehicle" data-vin="{VIN}">
      <img src="/images/no-photo-placeholder.png" alt="No photo available">
    </main><section><h2>Why Buy Used Vehicles from Us</h2>
      <img src="{logo}" alt="Jim Norton Toyota Logo">
    </section></body></html>
    """

    result = extract_vdp(
        html,
        detail_url=URL,
        origin="https://dealer.example",
        detail=DetailSpec(root_selector="main.vehicle", fields={}),
        expected_vin=VIN,
    )

    assert result.identity_proven
    assert result.photos == ()
    assert "photos" not in result.record


def test_gallery_survives_stray_decoration_and_duplicate_render() -> None:
    """A lone gallery-classed decoration or a responsive duplicate must not
    make the real multi-photo gallery ambiguous; two genuinely different
    multi-photo regions still fail closed."""

    from weaver.vehicle.vdp import _gallery_containers
    from bs4 import BeautifulSoup

    vin = "1GC4YNEY6MF193540"
    photos = "".join(
        f'<div class="vehicleImages col-md-6"><img src="https://cdn.example/{vin}-{i}.jpg"></div>'
        for i in range(9)
    )
    base = f"""
    <div id="root" data-vin="{vin}">
      <div class="oem-gallery">
        <div class="gallery-hero-img"><img src="https://cdn.example/{vin}-hero.jpg"></div>
        {photos}
      </div>
      <div class="images my-2 flex-1"><img src="https://cdn.example/dealer-badge.jpg"></div>
    </div>
    """
    scope = BeautifulSoup(base, "html.parser")
    containers = _gallery_containers(scope, None, vin)
    assert containers is not None and len(containers) == 1
    assert len(containers[0].find_all("img")) == 10

    duplicate = f"""
    <div id="root" data-vin="{vin}">
      <div class="oem-gallery d-none d-md-block">
        {photos}
        <div class="gallery-hero-img"><img src="https://cdn.example/{vin}-hero.jpg"></div>
      </div>
      <div class="mobile-gallery d-md-none">
        <img src="https://cdn.example/{vin}-hero.jpg">
        <img src="https://cdn.example/{vin}-0.jpg">
      </div>
    </div>
    """
    scope = BeautifulSoup(duplicate, "html.parser")
    containers = _gallery_containers(scope, None, vin)
    assert containers is not None and len(containers) == 1
    assert len(containers[0].find_all("img")) == 10

    rail = f"""
    <div id="root" data-vin="{vin}">
      <div class="oem-gallery">
        <img src="https://cdn.example/{vin}-hero.jpg">
        <img src="https://cdn.example/{vin}-0.jpg">
      </div>
      <div class="secondary-photo-carousel">
        <img src="https://cdn.example/other-1.jpg">
        <img src="https://cdn.example/other-2.jpg">
      </div>
    </div>
    """
    scope = BeautifulSoup(rail, "html.parser")
    assert _gallery_containers(scope, None, vin) is None


def test_cdn_prefix_gallery_scoops_own_folder_only() -> None:
    """Comma-joined script galleries bind by the og:image folder prefix; a
    similar-vehicles rail in another car's folder is never scooped."""

    vin = "1GC4YNEY6MF193540"
    own = ",".join(
        f"https://content.homenetiol.com/2000157/2065512/0x0/photo{i}.jpg"
        for i in range(10)
    )
    foreign = ",".join(
        f"https://content.homenetiol.com/2000157/9999999/0x0/other{i}.jpg"
        for i in range(4)
    )
    html = f"""
    <html><head>
      <link rel="canonical" href="https://dealer.example/viewdetails/used/{vin.lower()}/x">
      <meta property="og:image" content="https://content.homenetiol.com/2000157/2065512/0x0/photo0.jpg">
    </head><body>
      <main class="vehicle" data-vin="{vin}"><h1>2021 Chevrolet Silverado</h1></main>
      <script>var gallery = "{own}";</script>
      <script>var similar = "{foreign},https://content.homenetiol.com/2000157/2065512/nophoto/placeholder.jpg";</script>
    </body></html>
    """
    detail = DetailSpec(root_selector="main.vehicle", fields={})
    result = extract_vdp(
        html,
        detail_url=f"https://dealer.example/viewdetails/used/{vin.lower()}/x",
        origin="https://dealer.example",
        detail=detail,
        expected_vin=vin,
    )
    assert result.identity_proven
    urls = [photo.url for photo in result.photos]
    assert len([u for u in urls if "/2065512/" in u]) == 10
    assert not any("/9999999/" in u for u in urls)
    assert not any("nophoto" in u for u in urls)


def test_cdn_prefix_gallery_requires_cdn_anchored_primary() -> None:
    vin = "1GC4YNEY6MF193540"
    blob = ",".join(
        f"https://content.homenetiol.com/2000157/2065512/0x0/photo{i}.jpg"
        for i in range(6)
    )
    html = f"""
    <html><head>
      <meta property="og:image" content="https://dealer.example/static/hero.jpg">
    </head><body>
      <main class="vehicle" data-vin="{vin}"></main>
      <script>var gallery = "{blob}";</script>
    </body></html>
    """
    result = extract_vdp(
        html,
        detail_url=f"https://dealer.example/viewdetails/used/{vin.lower()}/x",
        origin="https://dealer.example",
        detail=DetailSpec(root_selector="main.vehicle", fields={}),
        expected_vin=vin,
    )
    assert not any("/2065512/" in photo.url for photo in result.photos)


def test_stock_render_only_vdp_is_a_corroborated_no_photo_exception() -> None:
    """A car whose only imagery is a manufacturer stock render carries zero
    photos and the placeholder corroboration, never a one-photo gallery."""

    vin = "3N8AP6CA5SL359104"
    stock = "https://content.homenetiol.com/2000157/2065512/0x0/stock_images/5/2025NIS26_640/x_640_01.jpg"
    html = f"""
    <html><head>
      <meta property="og:image" content="{stock}">
    </head><body>
      <main class="vehicle" data-vin="{vin}"><h1>2025 Nissan Kicks</h1></main>
    </body></html>
    """
    result = extract_vdp(
        html,
        detail_url=f"https://dealer.example/viewdetails/used/{vin.lower()}/x",
        origin="https://dealer.example",
        detail=DetailSpec(root_selector="main.vehicle", fields={}),
        expected_vin=vin,
    )
    assert result.identity_proven
    assert result.photos == ()
    assert result.placeholder_photo_published


def test_renditions_of_one_photo_are_one_photo() -> None:
    """A CDN publishes the same image at several sizes. Counting those as
    separate photos let a one-photo car satisfy the two-photo publishing
    contract — 43 of Orlando Nissan's 289 live vehicles were listed with the
    same picture twice (2026-08-29)."""

    from weaver.vehicle.vdp import PhotoEvidence, _dedupe_photos, photo_asset_key

    original = "https://assets.cai-media-management.com/common-vehicle-media/fee37d3d.jpg"
    resized = "https://assets.cai-media-management.com/resize/1024x1024/common-vehicle-media/fee37d3d.jpg"
    assert photo_asset_key(original) == photo_asset_key(resized)

    deduped = _dedupe_photos(
        [
            PhotoEvidence(resized, "social_meta", width=1024, full_resolution_candidate=True),
            PhotoEvidence(original, "data_full", width=None, full_resolution_candidate=True),
        ],
        40,
    )
    assert len(deduped) == 1
    # The un-resized original is what survives, not the social thumbnail.
    assert deduped[0].url == original

    # Genuinely different photos in the same sized folder stay separate.
    first = "https://content.homenetiol.com/2000157/2065512/0x0/aaa.jpg"
    second = "https://content.homenetiol.com/2000157/2065512/0x0/bbb.jpg"
    assert photo_asset_key(first) != photo_asset_key(second)
    assert len(_dedupe_photos(
        [
            PhotoEvidence(first, "data_full", width=1600, full_resolution_candidate=True),
            PhotoEvidence(second, "data_full", width=1600, full_resolution_candidate=True),
        ],
        40,
    )) == 2
    # The per-vehicle folder must survive folding, or two cars' photos merge.
    assert "2065512" in photo_asset_key(first)


def test_a_vin_in_the_photo_path_proves_ownership() -> None:
    """Post Oak Toyota's VDP carries 166 images and looked photoless, because
    its CDN host was unknown. Several inventory CDNs file a car's photos under
    its VIN, which is stronger proof than any folder convention — and immune
    to the same car being served from more than one shard."""

    from bs4 import BeautifulSoup

    from weaver.vehicle.vdp import _vin_path_gallery_candidates

    vin = "1N4BL4DV4SN384471"
    other = "5XYZT3LB3HG405174"
    html = (
        "<html><body>"
        f'<script>var g="https://vehicle-images.carscommerce.inc/686c-11001492/{vin}/aaa.png,'
        f'https://vehicle-images.carscommerce.inc/e00a-11001492/{vin}/bbb.png";</script>'
        f'<img src="https://vehicle-images.carscommerce.inc/686c-11001492/{vin}/thumbnails/large/aaa.png">'
        f'<img src="https://vehicle-images.carscommerce.inc/686c-11001492/{other}/not-mine.png">'
        f'<img src="https://cdn.example/stock_images/{vin}/factory-render.png">'
        "</body></html>"
    )
    soup = BeautifulSoup(html, "html.parser")
    candidates = _vin_path_gallery_candidates(soup, vin)
    assert candidates, "a VIN-named gallery must be recognised"
    images = candidates[0].value["images"]

    assert len(images) == 2
    assert all(vin in url for url in images)
    assert not any(other in url for url in images)          # another car's photos
    assert not any("thumbnail" in url for url in images)     # a rendition, not a photo
    assert not any("stock_images" in url for url in images)  # factory art, not this car
    # Two CDN shards serve the same car; both are still its photos.
    assert len({url.split("/")[3] for url in images}) == 2

    # Without a real VIN there is nothing to prove ownership against.
    assert _vin_path_gallery_candidates(soup, None) == []
    assert _vin_path_gallery_candidates(soup, "URLKEY12345678901") == []
