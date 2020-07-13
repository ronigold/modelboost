from distutils.core import setup
setup(
  name = 'modelboost',   
  packages = ['modelboost'],  
  version = '0.0.1',      
  license='MIT',      
  description = 'A package that makes the process of finding the model and optimizing the model automated and simple',
  author = 'Roni Gold',                  
  author_email = 'ronigoldsmid@gmail.com', 
  url = 'https://github.com/ronigold',
  download_url = 'https://github.com/user/reponame/archive/v_01.tar.gz',    # I explain this later on
  keywords = ['machine learning', 'deep learning', 'model', 'optimizing'],  
  install_requires=[            # I get to this in a second
          'pandas',
          'sklearn',
		  'IPython',
		  'tqdm',
		  'catboost',
		  'xgboost',
		  'lightgbm'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers', 
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License', 
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)