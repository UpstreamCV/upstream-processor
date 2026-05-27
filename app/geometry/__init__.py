"""P6-C — operator-saved geometry consumed from the eligible-session envelope.

Two halves:

- ``session_geometry``: pydantic models that mirror the api-server's
  ``GeometryConfig`` contract (normalized ``[0,1]`` coords, slug-ID
  validation, polygon/line cardinality constraints). The processor parses
  the wire envelope through these so a malformed config from the api is
  caught locally instead of producing nonsense pixel coordinates.

- ``resolver``: converts that parsed envelope into the runtime's
  ``AnalyticsGeometry`` (pixel coords, the shape ``build_analytics_config``
  consumes) plus a ``ResolvedGeometry`` wrapper carrying lineage
  (``source``, ``revision``) for status/observation tagging.
"""
