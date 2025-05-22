from fastapi.responses import JSONResponse

def handler_error(error_code, error_msg):
    response = {'error_code': error_code, "message": error_msg},
    return JSONResponse(response, status_code = error_code)