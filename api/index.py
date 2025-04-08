def handler(request):
    return {
        "statusCode": 200,
        "headers": { "Content-Type": "application/json" },
        "body": '{"message": "GDM API is live!"}'
    }
