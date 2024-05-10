class NodeStatus():
    def __init__(self, ip='127.0.0.1',
                    listen_port=None,
                    status=None,
                    last_ping_timestamp=None,
                    last_pong_timestamp=None,
                    last_pfail_timestamp=None,
                    fail_reports=None
                    ):
        self.ip = ip
        self.listen_port = listen_port
        self.status = status
        self.last_ping_timestamp = last_ping_timestamp
        self.last_pong_timestamp = last_pong_timestamp
        self.last_pfail_timestamp = last_pfail_timestamp
        self.failure_report_list = {}


        