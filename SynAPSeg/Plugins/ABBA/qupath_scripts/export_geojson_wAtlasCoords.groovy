//import qupath.ext.biop.warpy.Warpy
import net.imglib2.RealPoint
import qupath.lib.measurements.MeasurementList
import qupath.ext.biop.abba.AtlasTools
import static qupath.lib.gui.scripting.QPEx.* // For intellij editor autocompletion

setImageType('FLUORESCENCE');
qupath.ext.biop.abba.AtlasTools.loadWarpedAtlasAnnotations(getCurrentImageData(), "acronym", true);

/**
 * exports all detections (region annotations) as a geojson file, which is used in ABBA_PQA quantification 
    to localize nuclei to regions and must be run to use the ABBA_PQA pipeline
 * Also computes the centroid coordinates of each detection (region annotation)
    then adds these coordinates onto the measurement list. 
 * Measurements names: "Atlas_X", "Atlas_Y", "Atlas_Z"
 */

def pixelToAtlasTransform = 
    AtlasTools
    .getAtlasToPixelTransform(getCurrentImageData())
    .inverse() // pixel to atlas = inverse of atlas to pixel

// for each annotation get atlas coordinate at centroid
getAnnotationObjects().forEach(detection -> {
    RealPoint atlasCoordinates = new RealPoint(3);
    MeasurementList ml = detection.getMeasurementList();
    atlasCoordinates.setPosition([detection.getROI().getCentroidX(),detection.getROI().getCentroidY(),0] as double[]);
    pixelToAtlasTransform.apply(atlasCoordinates, atlasCoordinates);
//    ml.putMeasurement("Atlas_X", atlasCoordinates.getDoublePosition(0) )
//    ml.putMeasurement("Atlas_Y", atlasCoordinates.getDoublePosition(1) )
//    ml.putMeasurement("Atlas_Z", atlasCoordinates.getDoublePosition(2) )
})

// export all annotations
geoJsonOutdir = buildFilePath(PROJECT_BASE_DIR, "qupath_export_geojson")
mkdirs(geoJsonOutdir)
thisImgName = GeneralTools.getNameWithoutExtension(getCurrentImageData().getServer().getMetadata().getName())
thisOutName = buildFilePath(geoJsonOutdir, thisImgName+".geojson")
exportAllObjectsToGeoJson(thisOutName, "FEATURE_COLLECTION")