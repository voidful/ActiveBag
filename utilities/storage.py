from nlp2 import *


class Storage:
    __training_file = ''
    __validation_file = ''
    __testing_file = ''
    __output_loc = ''
    __retrain_file = ''

    def __init__(self, storage_params, identify_id_length=4):
        self.markId = random_string_with_timestamp(identify_id_length)
        try:
            if 'id' in storage_params:
                self.markId = storage_params['id']
                storage_params.update(self.load_storage_param())

            if 'retrain_file' in storage_params:
                self.set_retrain_file(storage_params['retrain_file'])

            self.set_training_file(storage_params['training_file'])
            self.set_validation_file(storage_params['validation_file'])
            self.set_testing_file(storage_params['testing_file'])
            self.set_output_dir(storage_params['output_dir'])
            self.write_storage_param(storage_params)
        except:
            raise AttributeError("Parameter Missing")

    def set_training_file(self, loc):
        if not is_file_exist(loc):
            raise Exception('file not found')
        self.__training_file = loc

    def set_retrain_file(self, loc):
        if not is_file_exist(loc):
            raise Exception('file not found')
        self.__retrain_file = loc

    def set_testing_file(self, loc):
        if not is_file_exist(loc):
            raise Exception('file not found')
        self.__testing_file = loc
        return loc

    def set_validation_file(self, loc):
        if not is_file_exist(loc):
            raise Exception('file not found')
        self.__validation_file = loc

    def set_output_dir(self, output_dir):
        output_dir += self.markId + "/"
        self.__output_loc = output_dir
        return get_dir_with_notexist_create(output_dir)

    def tidy_retrain_data(self):
        retrain_data = read_files_into_lines(self.get_retrain_file())
        with open(self.__training_file, "a", encoding='utf8') as tf:
            tf.write("\n".join(retrain_data))
            tf.write("\n")
        with open(self.__validation_file, "r+", encoding='utf8') as vf:
            old_valid = vf.readlines()
            vf.seek(0)
            for line in old_valid:
                if line.strip() not in retrain_data:
                    vf.write(line)
            vf.truncate()
        pass

    def write_classifier_param(self, param):
        output_filename = "/classifier_param.json"
        return write_json_to_file(param, self.__output_loc + output_filename)

    def load_classifier_param(self):
        output_filename = "/classifier_param.json"
        with open(self.__output_loc + output_filename) as f:
            return json.loads(f.read())

    def write_storage_param(self, param):
        output_filename = "/storage_param.json"
        return write_json_to_file(param, self.__output_loc + output_filename)

    def load_storage_param(self):
        output_filename = "/storage_param.json"
        with open(self.__output_loc + output_filename) as f:
            return json.loads(f.read())

    def get_training_file(self):
        return self.__training_file

    def get_retrain_file(self):
        return self.__retrain_file

    def get_testing_file(self):
        return self.__testing_file

    def get_validation_file(self):
        return self.__validation_file

    def get_classifier_dir(self):
        return get_dir_with_notexist_create(self.__output_loc + "classifiers/")

    def get_output_dir(self):
        return get_dir_with_notexist_create(self.__output_loc)

    def save_training_result(self):
        pass

    def list_all_classifier(self):
        return list(get_files_from_dir(self.get_classifier_dir()))

    def remove_classifier(self, loc):
        os.remove(loc)
