# Websocket policy transport

This directory contains the small, model-agnostic websocket transport used by the SIV-WAM policy server.

- `websocket_policy_server.py`: wraps a policy with a websocket server.
- `websocket_client_policy.py`: client-side policy adapter.
- `msgpack_numpy.py`: NumPy-aware message packing.
- `image_tools.py`: image conversion helpers.

The transport does not contain checkpoints or a specific model implementation. Pass an object exposing an `infer(observation)` method to the server wrapper and configure the host/port from the launch script.
