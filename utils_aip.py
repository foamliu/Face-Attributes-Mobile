import base64

from aip.face import AipFace


def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()


def get_face_attributes(filePath):
    appId = '16217748'
    apiKey = 'fq8w67yHRje2Tmr1UKhD9lCs'
    secretKey = 'srTGstYbXL4cDmbGM8iVF7amyqFbxPfN'

    aipFace = AipFace(appId, apiKey, secretKey)

    options = {}
    options["face_field"] = "age,beauty,expression,faceshape,gender,glasses,race,quality,facetype"
    options["max_face_num"] = 1
    options["face_type"] = "LIVE"

    image = base64.b64encode(get_file_content(filePath)).decode()
    result = aipFace.detect(image=image, image_type="BASE64", options=options)
    attr = {}
    if result is not None and 'result' in result and result['result'] is not None:
        face = result['result']['face_list'][0]
        attr['location'] = face['location']
        attr['face_probability'] = face['face_probability']
        attr['angle'] = face['angle']
        attr['age'] = face['age']
        attr['beauty'] = face['beauty']
        attr['expression'] = face['expression']
        attr['face_shape'] = face['face_shape']
        attr['gender'] = face['gender']
        attr['glasses'] = face['glasses']
        attr['race'] = face['race']
        attr['quality'] = face['quality']
        attr['face_type'] = face['face_type']
    return attr
