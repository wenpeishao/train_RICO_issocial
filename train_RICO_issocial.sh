#!/bin/bash

# Specify the environment name and staging path
ENVNAME=rico-analysis
STAGING_PATH=/staging/groups/jacobucci_group

# Print the current working directory for verification
echo "Current directory: $(pwd)"

# Copy the packed Conda environment from staging (update if you have a different environment file)
echo "Copying Conda environment from /staging..."
cp $STAGING_PATH/rico-analysis.tar.gz ./

# Set the environment directory name (same as the ENVNAME unless specified otherwise)
export ENVDIR=$ENVNAME

# Handle setting up the environment path and activating it
echo "Setting up Conda environment..."
mkdir -p $ENVDIR
tar -xzf rico-analysis.tar.gz -C $ENVDIR

# Activate the environment using an explicit path to Conda's activate script
echo "Activating Conda environment..."
source $ENVDIR/bin/activate  # Use 'source' instead of '.'

# Verify that the environment is activated and check the PATH variable
echo "Environment activated. Current PATH: $PATH"
which python  # This should show the path to the python executable inside the conda environment
python --version  # Verify the Python version to ensure it's from the Conda environment

# If 'which python' fails, attempt to manually add Python to PATH
if ! which python; then
    echo "Python not found after activation. Manually adding to PATH."
    export PATH=$ENVDIR/bin:$PATH
    echo "Updated PATH: $PATH"
    which python
fi

# Copy necessary dataset files from /staging to the working directory
echo "Copying dataset files from /staging..."
cp $STAGING_PATH/unique_uis.tar.gz ./
cp $STAGING_PATH/ui_details_updated.csv ./
cp $STAGING_PATH/ui_layout_vectors.zip ./
cp $STAGING_PATH/traces.tar.gz ./

# Optionally, copy additional files if needed
# cp $STAGING_PATH/animations.tar.gz ./
# cp $STAGING_PATH/app_details.csv ./
# cp $STAGING_PATH/semantic_annotations.zip ./

# Unzip and untar the dataset files into the local working directory
echo "Extracting dataset files..."
mkdir -p combined/
tar -xzf unique_uis.tar.gz -C combined/
tar -xzf traces.tar.gz -C combined/
unzip -o ui_layout_vectors.zip -d combined/

# Remove zip files to free up space
echo "Removing compressed files to free up space..."
rm -f unique_uis.tar.gz traces.tar.gz ui_layout_vectors.zip

# Verify the extracted contents
echo "Listing contents of the combined directory..."
ls -la combined/

# Run your Python script for processing RICO dataset (replace with your script name)
echo "Running the RICO analysis script..."
python3 train_RICO_issocial.py

# Clean up any large files after the job completes to avoid unnecessary file transfers
echo "Cleaning up extracted files..."
rm -rf combined/

# Deactivate and clean up the environment if necessary
echo "Deactivating environment and cleanup..."
conda deactivate
rm -rf $ENVDIR
echo "Environment deactivated and cleaned up."
