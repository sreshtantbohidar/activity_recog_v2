import ast
import unittest
from pathlib import Path

from swagger_docs import (
    ANALYZE_IMAGE_UPLOAD_SWAGGER_DOC,
    ANALYZE_VIDEO_UPLOAD_SWAGGER_DOC,
    ANALYZE_VIDEO_DETAILED_SWAGGER_DOC,
    DETECT_OBJECTS_SWAGGER_DOC,
    DOWNLOAD_REPORT_SWAGGER_DOC,
    GET_CONFIG_SWAGGER_DOC,
    SET_CONFIG_SWAGGER_DOC,
    FETCH_MEDIA_SWAGGER_DOC,
    GET_QUEUE_SWAGGER_DOC,
    GET_QUEUE_JOB_SWAGGER_DOC,
    DELETE_QUEUE_JOB_SWAGGER_DOC,
    CLEAR_QUEUE_SWAGGER_DOC,
    CANCEL_QUEUE_JOB_SWAGGER_DOC,
    RELOAD_QUEUE_JOB_SWAGGER_DOC,
    GET_VISION_BACKEND_SWAGGER_DOC,
    HEALTH_SWAGGER_DOC,
    JOB_STATUS_VALUES,
    SET_VISION_BACKEND_SWAGGER_DOC,
)


class SwaggerOperationDocsTests(unittest.TestCase):
    def test_all_api_operations_have_summaries_and_responses(self):
        for operation in (
            HEALTH_SWAGGER_DOC,
            FETCH_MEDIA_SWAGGER_DOC,
            DETECT_OBJECTS_SWAGGER_DOC,
            DOWNLOAD_REPORT_SWAGGER_DOC,
            GET_CONFIG_SWAGGER_DOC,
            SET_CONFIG_SWAGGER_DOC,
            ANALYZE_IMAGE_UPLOAD_SWAGGER_DOC,
            ANALYZE_VIDEO_UPLOAD_SWAGGER_DOC,
            ANALYZE_VIDEO_DETAILED_SWAGGER_DOC,
            GET_VISION_BACKEND_SWAGGER_DOC,
            SET_VISION_BACKEND_SWAGGER_DOC,
            GET_QUEUE_SWAGGER_DOC,
            GET_QUEUE_JOB_SWAGGER_DOC,
            DELETE_QUEUE_JOB_SWAGGER_DOC,
            CLEAR_QUEUE_SWAGGER_DOC,
            CANCEL_QUEUE_JOB_SWAGGER_DOC,
            RELOAD_QUEUE_JOB_SWAGGER_DOC,
        ):
            with self.subTest(summary=operation["summary"]):
                self.assertTrue(operation["summary"])
                self.assertIn("responses", operation)
                # Queued endpoints answer 202; the inline ones still answer 200.
                self.assertTrue({"200", "202"} & set(operation["responses"]), operation["responses"])

    def test_health_documents_degraded_backend_response(self):
        self.assertIn("503", HEALTH_SWAGGER_DOC["responses"])

    def test_media_apis_document_duration_and_failure_modes(self):
        # /fetch-media still runs inline, so its inference failure is a 502.
        self.assertIn("413", FETCH_MEDIA_SWAGGER_DOC["responses"])
        self.assertIn("502", FETCH_MEDIA_SWAGGER_DOC["responses"])
        self.assertEqual(FETCH_MEDIA_SWAGGER_DOC["parameters"][0]["schema"]["required"], ["id"])
        # /detect-objects is queued: it accepts, rejects bad input, or sheds load.
        for code in ("202", "400", "413", "503"):
            self.assertIn(code, DETECT_OBJECTS_SWAGGER_DOC["responses"])

    def test_queued_endpoints_document_acceptance_and_capacity(self):
        for operation in (ANALYZE_IMAGE_UPLOAD_SWAGGER_DOC, ANALYZE_VIDEO_UPLOAD_SWAGGER_DOC,
                          ANALYZE_VIDEO_DETAILED_SWAGGER_DOC, DETECT_OBJECTS_SWAGGER_DOC):
            self.assertIn("202", operation["responses"])
            self.assertIn("503", operation["responses"])
            # A queued job cannot fail synchronously with the backend in reach.
            self.assertNotIn("502", operation["responses"])

    def test_queue_apis_document_pagination_filters_and_lookup(self):
        parameters = {parameter["name"]: parameter for parameter in GET_QUEUE_SWAGGER_DOC["parameters"]}
        self.assertEqual(set(parameters), {"page", "page_size", "status", "include_response"})
        self.assertEqual(parameters["page"]["default"], 1)
        # The ceiling has to match what the route will actually serve.
        self.assertEqual(parameters["page_size"]["default"], 25)
        self.assertIn("1-200", parameters["page_size"]["description"])
        self.assertEqual(parameters["status"]["enum"], JOB_STATUS_VALUES)
        self.assertIn("400", GET_QUEUE_SWAGGER_DOC["responses"])
        self.assertIn("404", GET_QUEUE_JOB_SWAGGER_DOC["responses"])

    def test_queue_mutation_apis_document_their_guards(self):
        self.assertEqual(DELETE_QUEUE_JOB_SWAGGER_DOC["parameters"][0]["name"], "job_id")
        # A pending job is refused rather than cancelled, and the docs say so.
        self.assertIn("409", DELETE_QUEUE_JOB_SWAGGER_DOC["responses"])
        self.assertIn("404", DELETE_QUEUE_JOB_SWAGGER_DOC["responses"])
        self.assertNotIn("parameters", CLEAR_QUEUE_SWAGGER_DOC)
        self.assertIn("never", CLEAR_QUEUE_SWAGGER_DOC["description"])

    def test_queue_cancel_and_reload_document_their_outcomes(self):
        for operation in (CANCEL_QUEUE_JOB_SWAGGER_DOC, RELOAD_QUEUE_JOB_SWAGGER_DOC):
            self.assertEqual(operation["parameters"][0]["name"], "job_id")
            self.assertIn("404", operation["responses"])
            self.assertIn("409", operation["responses"])
        # Cancelling is cooperative, and the docs say so rather than implying a
        # running model call can be interrupted mid-flight.
        self.assertIn("checkpoint", CANCEL_QUEUE_JOB_SWAGGER_DOC["description"])
        # A reload is a new job, so it is accepted rather than completed inline.
        self.assertIn("202", RELOAD_QUEUE_JOB_SWAGGER_DOC["responses"])
        self.assertIn("bounded", RELOAD_QUEUE_JOB_SWAGGER_DOC["description"])

    def test_upload_apis_document_multipart_file_parameter(self):
        for operation in (ANALYZE_IMAGE_UPLOAD_SWAGGER_DOC, ANALYZE_VIDEO_UPLOAD_SWAGGER_DOC,
                          ANALYZE_VIDEO_DETAILED_SWAGGER_DOC):
            self.assertEqual(operation["consumes"], ["multipart/form-data"])
            file_param = operation["parameters"][0]
            self.assertEqual(file_param["name"], "file")
            self.assertEqual(file_param["in"], "formData")
            self.assertEqual(file_param["type"], "file")
            self.assertTrue(file_param["required"])

    def test_backend_switch_documents_input_and_failure_modes(self):
        body = SET_VISION_BACKEND_SWAGGER_DOC["parameters"][0]
        self.assertEqual(body["in"], "body")
        self.assertEqual(body["schema"]["required"], ["backend"])
        self.assertEqual(body["schema"]["properties"]["backend"]["enum"], ["vllm"])
        self.assertIn("400", SET_VISION_BACKEND_SWAGGER_DOC["responses"])
        self.assertIn("502", SET_VISION_BACKEND_SWAGGER_DOC["responses"])
        self.assertIn("available_backends", GET_VISION_BACKEND_SWAGGER_DOC["responses"]["200"]["schema"]["properties"])

    def test_every_flask_route_has_swagger_documentation(self):
        source = Path(__file__).with_name("app.py").read_text()
        tree = ast.parse(source)
        api_routes = {}
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            route_paths = []
            has_swagger = False
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                decorator_name = (
                    decorator.func.id if isinstance(decorator.func, ast.Name)
                    else decorator.func.attr if isinstance(decorator.func, ast.Attribute)
                    else None
                )
                if decorator_name == "route" and decorator.args:
                    if isinstance(decorator.args[0], ast.Constant):
                        route_paths.append(decorator.args[0].value)
                elif decorator_name == "swag_from":
                    has_swagger = True
            for route_path in route_paths:
                api_routes[route_path] = has_swagger

        self.assertEqual(
            set(api_routes),
            {
                "/",
                "/health",
                "/health/components",
                "/unit-tests",
                "/fetch-media",
                "/detect-objects",
                "/analyze-image",
                "/analyze-video",
                "/analyze-video-detailed",
                "/vision-backend",
                "/reports/<path:filename>",
                "/config",
                "/config/credentials",
                "/config/credentials/test",
                "/queue",
                "/queue/<job_id>",
                "/queue/<job_id>/cancel",
                "/queue/<job_id>/reload",
            },
        )
        self.assertTrue(all(documented for path, documented in api_routes.items() if path != "/"), api_routes)


if __name__ == "__main__":
    unittest.main()
