import os

from pathlib import Path
from joblib import dump, load


class Storage:
    def __init__(self, storage_dir: str):
        self.storage_dir = Path(storage_dir)
        if not self.storage_dir.exists():
            self.storage_dir.mkdir(parents=True)

    def save(self, model_obj, model_path: str, user_id: str = None, private: bool = False):
        if user_id is None and private:
            raise ValueError('Private saving need user_id')

        path_in_storage, model_name = self._get_path_in_storage_and_name(model_path)

        if private:
            model_storage_dir = self._get_private_storage_path(user_id) / path_in_storage
        else:
            model_storage_dir = self._get_public_storage_path() / path_in_storage

        if not model_storage_dir.exists():
            model_storage_dir.mkdir(parents=True, exist_ok=True)

        full_model_path = model_storage_dir / model_name

        dump(model_obj, full_model_path)

    def load(self, model_path: str, user_id: str = None, private=False):
        path_in_storage, model_name = self._get_path_in_storage_and_name(model_path)
        private_model_full_path = self._get_private_storage_path(user_id) / path_in_storage / model_name
        public_model_full_path = self._get_public_storage_path() / path_in_storage / model_name

        if public_model_full_path.exists() and not private:
            full_model_path = public_model_full_path
        elif user_id and private_model_full_path.exists():
            full_model_path = private_model_full_path
        else:
            raise ValueError(f'Model with path {model_path} not found')

        return load(full_model_path)

    def list(self, user_id):
        """
        Returns all models names from public storage and private if private=True
        """
        public_storage = self._get_public_storage_path()
        public_model_list = [
            (os.path.join(dp, f).replace(str(public_storage) + os.sep, ''), 'public')
            for dp, dn, filenames in os.walk(public_storage) for f in filenames
        ]
        private_storage = self._get_private_storage_path(user_id)
        private_model_list = [
            (os.path.join(dp, f).replace(str(private_storage) + os.sep, ''), 'private')
            for dp, dn, filenames in os.walk(private_storage) for f in filenames
        ]
        return public_model_list + private_model_list

    def _get_private_storage_path(self, user_id: str):
        return self.storage_dir / 'private' / user_id

    def _get_public_storage_path(self):
        return self.storage_dir / 'public'

    def delete(self, model_path: str, user_id, private=False):
        path_in_storage, model_name = self._get_path_in_storage_and_name(model_path)
        private_model_full_path = self._get_private_storage_path(user_id) / path_in_storage / model_name
        public_model_full_path = self._get_public_storage_path() / path_in_storage / model_name
        if private:
            private_model_full_path.unlink(missing_ok=True)
        else:
            public_model_full_path.unlink(missing_ok=True)

    @staticmethod
    def _get_path_in_storage_and_name(model_path: str):
        model_path = Path(model_path)
        model_name = model_path.name
        path_in_storage = str(model_path.parent)
        if path_in_storage[0] == os.sep:
            path_in_storage = path_in_storage[1:]
        return path_in_storage, model_name
