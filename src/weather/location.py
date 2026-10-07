"""Find the location to show weather for: city name, coordinates or IP address.

Nothing is hard-coded: the user types a place, enters coordinates, or asks for
an approximate location from the computer's public IP address.

    search_locations("Surat")          # list of matches (Open-Meteo geocoding, no key)
    from_coordinates(21.17, 72.83)
    locate_by_ip()                     # approximate; can be off by hundreds of km
"""

from dataclasses import dataclass

from .common import LocationError, get_json, malformed

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
IP_LOCATION_URL = "https://ipinfo.io/json"


@dataclass(frozen=True)
class Location:
    name: str
    latitude: float
    longitude: float
    region: str | None = None
    country: str | None = None
    source: str = "search"          # "search", "coordinates" or "ip"

    @property
    def label(self):
        return ", ".join(p for p in (self.name, self.region, self.country) if p)

    @property
    def coordinates(self):
        return f"{self.latitude:.4f}, {self.longitude:.4f}"


def validate_coordinates(latitude, longitude):
    try:
        lat, lon = float(latitude), float(longitude)
    except (TypeError, ValueError) as exc:
        raise LocationError("Latitude and longitude must be numbers.") from exc
    if not -90 <= lat <= 90:
        raise LocationError(f"Latitude must be between -90 and 90 (got {lat}).")
    if not -180 <= lon <= 180:
        raise LocationError(f"Longitude must be between -180 and 180 (got {lon}).")
    return lat, lon


def from_coordinates(latitude, longitude, name=None):
    lat, lon = validate_coordinates(latitude, longitude)
    return Location(name or f"{lat:.4f}, {lon:.4f}", lat, lon, source="coordinates")


def search_locations(query, count=5):
    """Places matching a name, best match first. Raises LocationError if none."""
    query = (query or "").strip()
    if len(query) < 2:
        raise LocationError("Enter a city or village name (at least 2 letters).")
    data = get_json(GEOCODING_URL, {"name": query, "count": count, "language": "en",
                                    "format": "json"}, "the location search service")
    try:
        results = [Location(r["name"], float(r["latitude"]), float(r["longitude"]),
                            r.get("admin1"), r.get("country"))
                   for r in data.get("results", [])]
    except (KeyError, TypeError, ValueError) as exc:
        raise malformed("the location search service", exc) from exc
    if not results:
        raise LocationError(f'No place called "{query}" was found. Check the spelling, '
                            "or enter latitude and longitude instead.")
    return results


def locate_by_ip():
    """Approximate location of this computer's public IP address (ipinfo.io).

    IP locations come from the internet provider and can be off by hundreds of
    kilometres; when the app runs on a server, it is the server's location.
    """
    data = get_json(IP_LOCATION_URL, {}, "the IP location service")
    try:
        lat, lon = (float(v) for v in data["loc"].split(","))
    except (KeyError, AttributeError, ValueError) as exc:
        raise LocationError("Your approximate location could not be determined. "
                            "Search for your city instead.") from exc
    lat, lon = validate_coordinates(lat, lon)
    return Location(data.get("city") or "Approximate location", lat, lon,
                    data.get("region"), data.get("country"), source="ip")
