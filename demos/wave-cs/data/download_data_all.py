#!/usr/bin/env python
"""
Downloads all data files from on-line storage using the URL and destination
folder specified in the comma-separated-varaiable './download_data_index.csv'
file, which should have the columns:

    filename,URL,folder,MD5

Specified paths are assumed to be relative to the detected location of
this python script, i.e., the location of download_data_all.py

If the MD5 hash of the downloaded file matches, the function increments the
reported SUCCESS_COUNT otherwise it increments the reported FAIL_COUNT

If a file is already downloaded, only the MD5 checksum is performed.

Warning messages are generated when:
       * file already exists locally
       * file did not download
       * file MD5 hash value does not match expected value

   If the file storage service changes, the URLs in the index file
   will be updated to reflect the new location.
"""
import os
import urllib
import hashlib

# detect location of this script
pathstr_python_script = os.path.dirname(os.path.realpath(__file__))

# full path to download_data_index.csv
filename_index = os.path.join(pathstr_python_script, 'download_data_index.csv')

# initialize counts
success_count = 0
fail_count = 0


def file_is_verified(destination_filename_fullpath, md5_expected, should_print_output):
    # Reports success if file exists on file-system and MD5 matches.
    result = False
    if os.path.isfile(destination_filename_fullpath):
        if should_print_output:
            print("%s exists" % (destination_filename_fullpath) )
        # success only if MD5 checksum matches
        md5_download = hashlib.md5(open(destination_filename_fullpath, 'rb').read()).hexdigest()
        if md5_expected==md5_download:
            if should_print_output:
                print("Success! MD5 checksum verified for %35s : %s (expected) == %s (downloaded)" % (filename_download, md5_expected, md5_download) )
            result = True
        else:
            if should_print_output:
                print("INVALID MD5 checksum for            %35s: %s (expected) != %s (downloaded)" % (filename_download, md5_expected, md5_download) )
            result = False
    return result


with open(filename_index, 'r') as f:
    lines = f.readlines()
    # loop over files. Skip header row of CSV.
    for line in lines[1:]:
        print("")
        # download details
        filename_download, URL, folder_relative_path, md5_expected = line.strip().split(',')
        destination_filename_fullpath = os.path.join(pathstr_python_script, folder_relative_path, filename_download)

        if file_is_verified(destination_filename_fullpath, md5_expected, True):
            print("Skipping download of verified file: %s" % destination_filename_fullpath)
            success_count += 1
        else:
            # download file
            print("Downloading file %s from %s" % (filename_download, URL) )
            filehandle = urllib.URLopener()
            filehandle.retrieve(URL, destination_filename_fullpath)

            if file_is_verified(destination_filename_fullpath, md5_expected, True):
                success_count += 1
            else:
                fail_count += 1
                print("%s did not successfully download" % (filename_download) )

print("\nSuccessfully downloaded and verified %.2f%% : success_count = %d, fail_count = %d\n" % (100.0 * success_count/float(success_count+fail_count), success_count, fail_count) )
