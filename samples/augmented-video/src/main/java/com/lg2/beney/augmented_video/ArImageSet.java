package com.lg2.beney.augmented_video;

import androidx.annotation.NonNull;

import java.util.List;
import java.util.Optional;

public class ArImageSet {

    public static final String LOG_TAG = ArImageSet.class.getSimpleName();

    public ArImageInfo RepArImage;

    public List<ArImageInfo> ArImages;

    public String Name;

    public ArImageInfo FindByHash(long hashCode)
    {
        try
        {
            if (ArImages.size() == 0)
            {
                return null;
            }

            Optional<ArImageInfo> target = ArImages.stream().filter(x->x.IsSimilar(hashCode)).findFirst();
            return target.orElse(null);

        }
        catch (Exception ex)
        {
            return null;
        }
    }

    public boolean HasImage(long hashCode)
    {
        try
        {
            if (ArImages.size() == 0)
            {
                return false;
            }

            return ArImages.stream().anyMatch(x -> x.IsSimilar(hashCode));
        }
        catch (Exception ex)
        {
            return false;
        }

        // return FindByHash(hashCode) != null;
    }

    @Override
    public boolean equals(Object other)
    {
        if (other == null) return false;
        if (RepArImage == null) return false;

        if (!(other instanceof ArImageSet)) {
            return false;
        }

        ArImageSet rArImageSet = (ArImageSet) other;

        if (rArImageSet.RepArImage == null) return false;

        long rId = RepArImage.Hash;
        long lId = rArImageSet.RepArImage.Hash;

        return rId != lId;
    }

    @NonNull
    @Override
    public String toString()
    {
        StringBuilder sb = new StringBuilder();
        sb.append("{")
                .append("Name=").append(Name)
                .append(", RepArImage=").append(RepArImage)
                .append(", ArImages=").append(ArImages).append("}");
        return sb.toString();
    }
}
