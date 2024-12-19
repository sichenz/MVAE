
require(dplyr)

# This monster function creates the merged_feats object; it requires setting the working directory properly,
# and making sure the labels are loaded (i.e. it's not a very "good" function). It does minimal processing.
# Its functionality mirrors the "create_visual_dataframe.R" script.

create.merged.feats = function(name, colors, hulls, fonts, global_feats, marks, mark_feats){

  all_fonts <- read.csv("data/fonts_erosion-char_higher_fixed.csv")
  
  color_labels <- read.csv("data/FINAL_cluster_centers/colors_descs.csv")
  color_labels$desc <- as.character(color_labels$desc)
  
  hull_labels <- c("circle",
                   "rectange-oval_medium",
                   "rectangle-oval_large",
                   "rectangle-oval_thin",
                   "square",
                   "triangle")
  
  mark_orig_labels <- c("bad_letters",
                        "bulky_hollow_geometric",
                        "circular",
                        "dense_simple_geometric",
                        "detailed_fit_in_circle",
                        "hollow_circles",
                        "horizontal_complex",
                        "long_horizonal",
                        "simple_shapes",
                        "square",
                        "thin_vertical_rectangle",
                        "vertical_narrow",
                        "very_detailed",
                        "very_thin",
                        "wispy_horizontal_lines")
  
  colors$color_name <- color_labels$desc[match(colors$color_label, color_labels$cluster)]
  hulls$hull_name <- hull_labels[hulls$label + 1]
  
  if(!all(is.na(marks))){
    marks$orig_name <- mark_orig_labels[marks$cluster + 1]
    
    biggest_mark_feats <- do.call(rbind, by(mark_feats, mark_feats$name, function(x){x[which.max(x$frac),]}))
    rownames(biggest_mark_feats) <- biggest_mark_feats$name
    biggest_mark_feats <- biggest_mark_feats[,-which(colnames(biggest_mark_feats)=="name")]
    
    biggest_mark_cluster_name <- as.character(marks$orig_name)[which.max(marks$frac)]
    biggest_mark_feats$cluster_name <- biggest_mark_cluster_name
  } else {
    biggest_mark_cluster_name = NA
  }

  
  
  # Process fonts -------------------------------
  
  font_form_counts <- t(as.matrix(table(factor(fonts$ftype, levels=c("sans","serif","callig")))))
  colnames(font_form_counts) <- paste("form_", c("sans","serif","callig"), sep="")
  
  # Compute the number of letters in each logo:
  nletters <- sum(font_form_counts)
  
  ## The font classification, based on the Vox-ATypI classification scheme:
  font_class_counts <- table(factor(fonts$style, levels=unique(all_fonts$style)))
  font_class_counts["didone"] <- font_class_counts["didone"] + font_class_counts["kepler"]
  font_class_counts <- t(as.matrix(font_class_counts[-which(names(font_class_counts) == "kepler")]))
  colnames(font_class_counts) <- paste("class_", colnames(font_class_counts), sep="")
  
  ## Family is the specific font name, like Times New Roman:
  font_family_counts <- table(factor(fonts$family, levels=unique(all_fonts$family)))
  names(font_family_counts) <- paste("family_", names(font_family_counts), sep="")
  font_family_counts = t(as.matrix(font_family_counts))
  
  ## Version refers to the details, conditional on a font family: things like bold,
  ## italics, wide, condensed, etc.
  font_ver_counts <- table(factor(fonts$ver, levels=unique(all_fonts$ver)))
  names(font_ver_counts) <- paste("ver_", names(font_ver_counts), sep="")
  oldnames <- names(font_ver_counts)
  
  font_ver_counts <- as.data.frame(matrix(font_ver_counts, nrow=1))
  colnames(font_ver_counts) = oldnames
  
  # width vars:
  condensed_cols <- paste("ver_", c('cb','co','cbi','ci'), sep="")
  wide_cols <- paste("ver_", c('wide','wb','wbi','wi','wl','wli'), sep="")
  orig_width_cols <- setdiff(oldnames, c(condensed_cols, wide_cols))
  font_ver_counts$width_condensed <- rowSums(font_ver_counts[,colnames(font_ver_counts) %in% condensed_cols])
  font_ver_counts$width_wide <- rowSums(font_ver_counts[,colnames(font_ver_counts) %in% wide_cols])
  font_ver_counts$width_orig <- rowSums(font_ver_counts[,colnames(font_ver_counts) %in% orig_width_cols])
  font_ver_counts$width_has_condensed <- 1*(font_ver_counts$width_condensed/nletters > 0.25)
  font_ver_counts$width_has_wide <- 1*(font_ver_counts$width_wide/nletters > 0.25)
  font_ver_counts$width_has_orig <- 1*(font_ver_counts$width_orig/nletters > 0.25)
  font_ver_counts$width_mixed <- 1*(font_ver_counts$width_has_condensed*font_ver_counts$width_has_wide+ 
                                      font_ver_counts$width_has_condensed*font_ver_counts$width_has_orig + 
                                      font_ver_counts$width_has_orig*font_ver_counts$width_has_wide > 0)
  
  # weight vars:
  bold_cols <- paste("ver_", c('bold','cb','bi','cbi','wb','wbi'), sep="")
  light_cols <- paste("ver_", c('light','li','wl','wli'), sep="")
  orig_weight_cols <- setdiff(oldnames, c(bold_cols, light_cols))
  font_ver_counts$weight_bold <- rowSums(font_ver_counts[,colnames(font_ver_counts) %in% bold_cols])
  font_ver_counts$weight_light <- rowSums(font_ver_counts[,colnames(font_ver_counts) %in% light_cols])
  font_ver_counts$weight_orig <- rowSums(font_ver_counts[,colnames(font_ver_counts) %in% orig_weight_cols])
  font_ver_counts$weight_has_bold <- 1*(font_ver_counts$weight_bold/nletters > 0.25)
  font_ver_counts$weight_has_light <- 1*(font_ver_counts$weight_light/nletters > 0.25)
  font_ver_counts$weight_has_orig <- 1*(font_ver_counts$weight_orig/nletters > 0.25)
  font_ver_counts$weight_mixed <- 1*(font_ver_counts$weight_has_bold*font_ver_counts$weight_has_light + 
                                       font_ver_counts$weight_has_bold*font_ver_counts$weight_has_orig + 
                                       font_ver_counts$weight_has_orig*font_ver_counts$weight_has_light > 0)
  
  # style vars:
  italic_cols <- paste("ver_", c('bi','cbi','italic','ci','li','wi','wli','wbi'), sep="")
  orig_style_cols <- setdiff(oldnames, italic_cols)
  font_ver_counts$style_italic <- rowSums(font_ver_counts[,colnames(font_ver_counts) %in% italic_cols])
  font_ver_counts$style_orig <- rowSums(font_ver_counts[,colnames(font_ver_counts) %in% orig_style_cols])
  font_ver_counts$style_has_italic <- 1*(font_ver_counts$style_italic/nletters > 0.25)
  font_ver_counts$style_has_orig <- 1*(font_ver_counts$style_orig/nletters > 0.25)
  font_ver_counts$style_mixed <- font_ver_counts$style_has_italic * font_ver_counts$style_has_orig
  
  ## Now merge all of them to create a final font table:
  fonts_wide = cbind(font_form_counts, font_class_counts, font_family_counts, font_ver_counts)
  
  
  
  
  
  
  # Wide colors: ------------------------------------------------------------
  
  wide_colors <- t(as.matrix(table(factor(colors$color_name, levels=color_labels$desc))))
  
  wide_domcolor <- do.call(rbind, by(colors, colors$name,
                                     function(x){
                                       dcvec <- rep(0, ncol(wide_colors)-1)
                                       nonwhite <- x[x$color_name != "white",]
                                       dc <- nonwhite$color_label[which.max(nonwhite$frac)]
                                       dcvec[dc] <- 1
                                       dcvec
                                     }))
  
  colnames(wide_domcolor) <- paste("domcolor.", color_labels$desc[-1], sep="")
  
  domcolor <- do.call(c, by(colors, colors$name,
                            function(x){
                              nonwhite <- x[x$color_name != "white",]
                              dc <- nonwhite$color_label[which.max(nonwhite$frac)]
                              dc
                            },
                            simplify=F))
  
  domcolor_name <- color_labels$desc[-1][domcolor]
  names(domcolor_name) <- names(domcolor)
  
  
  
  # Wide hulls: -------------------------------------------------------------
  
  wide_hulls <- t(as.matrix(table(factor(hulls$hull_name, levels=hull_labels))))
  
  
  
  
  # Create merged global features data frame --------------------------------
  
  rownames(global_feats) <- global_feats$name; global_feats <- global_feats[,-which(colnames(global_feats)=="name")]
  colnames(global_feats) <- paste("global.", colnames(global_feats), sep="")
  
  global_feats$global.domcolor <- domcolor_name
  global_feats$global.hull_type <- hulls$hull_name
  global_feats$global.has_mark <- 1 - 1*all(is.na(marks))
  
  global_feats$global.bm_label <- biggest_mark_cluster_name
  
  colnames(fonts_wide) <- paste("font.", colnames(fonts_wide), sep="")
  colnames(wide_colors) <- paste("color.", colnames(wide_colors), sep="")
  
  merged_feats <- cbind(global_feats, fonts_wide, wide_colors)
  
  merged_feats
}

remove.prefix <- function(x, prefix_sep="[.]") {sapply(strsplit(x, prefix_sep), function(z) z[2])}




# This function is again not a very good function; it takes the merged_feats from above,
# and does further processing. Its functionality mirrors: (1) the "create_rdata.R" script, (2) the
# "data_prep.R" script from the model_free directory, and (3) the "reformat_real_feats.R" script,
# which discretizes the real values. It requires the cutoffs loaded below:



process.features = function(name, merged_feats){
  
  logo_feats <- merged_feats
  
  select.feature <- function(feature){
    sapply(strsplit(colnames(logo_feats), "[.]"), function(x) feature %in% x)
  }
  
  select.subfeature <- function(feature, subfeature){
    sapply(lapply(strsplit(colnames(logo_feats), c("[.]")), function(y) do.call(c, strsplit(y, "_"))), function(x) (feature %in% x) & (subfeature %in% x))
  }
  
  load("code/data_processing/quantile_cutoffs.RData")
  
  all_fonts <- read.csv("data/fonts_erosion-char_higher_fixed.csv")
  
  color_labels <- read.csv("data/FINAL_cluster_centers/colors_descs.csv")
  color_labels$desc <- as.character(color_labels$desc)
  
  hull_labels <- c("circle",
                   "rectange-oval_medium",
                   "rectangle-oval_large",
                   "rectangle-oval_thin",
                   "square",
                   "triangle")
  
  mark_orig_labels <- c("bad_letters",
                        "bulky_hollow_geometric",
                        "circular",
                        "dense_simple_geometric",
                        "detailed_fit_in_circle",
                        "hollow_circles",
                        "horizontal_complex",
                        "long_horizonal",
                        "simple_shapes",
                        "square",
                        "thin_vertical_rectangle",
                        "vertical_narrow",
                        "very_detailed",
                        "very_thin",
                        "wispy_horizontal_lines")
  
  domcolor_labels <- color_labels$desc[-which(color_labels$desc == "white")]
  domcolor <- model.matrix(~0+factor(logo_feats$global.domcolor, levels = domcolor_labels))
  colnames(domcolor) <- domcolor_labels
  
  colors <- logo_feats[, sapply(strsplit(colnames(logo_feats), "[.]"), function(x) x[1] == "color")]
  colnames(colors) <- remove.prefix(colnames(colors))
  colors <- colors[, -which(colnames(colors)=="white")]
  colors <- colors[, match(colnames(domcolor), colnames(colors))]
  
  accent_colors <- colors - domcolor
  
  genfont <- logo_feats[,colnames(logo_feats)[sapply(lapply(strsplit(colnames(logo_feats), "[.]"),
                                                            function(x) do.call(c, strsplit(x, "_"))),
                                                     function(x) any(all(c("font","has") %in% x), "form" %in% x, "mixed" %in% x))]]
  
  colnames(genfont) <- remove.prefix(colnames(genfont))
  
  
  
  fontclass <- logo_feats[, colnames(logo_feats)[sapply(lapply(strsplit(colnames(logo_feats), "[.]"),
                                                               function(x) do.call(c, strsplit(x, "_"))),
                                                        function(x) all(c("font","class") %in% x))]]
  
  colnames(fontclass) <- sapply(strsplit(colnames(fontclass), "_"), function(x) x[2])
  fontclass <- fontclass[,-which(colnames(fontclass)=="amp")]
  kept_classes = c("grotesque","geom","geom-square","humanist","transitional","oldstyle","clarendon","slab","didone")
  fontclass <- fontclass[,kept_classes]
  colnames(fontclass)[colnames(fontclass) == "geom-square"] <- "geom.square"
  
  bin_fontclass <- fontclass
  bin_fontclass[] <- 0
  bin_fontclass[cbind(1:nrow(bin_fontclass), apply(fontclass, 1, which.max))] <- 1
  
  hull_type <- model.matrix(~0+factor(logo_feats$global.hull_type, levels = hull_labels))
  colnames(hull_type) <- hull_labels
  
  
  logo_feats$global.bm_label <- as.character(logo_feats$global.bm_label)
  logo_feats$global.bm_label[is.na(logo_feats$global.bm_label)] <- "no_mark"
  logo_feats$global.bm_label <- as.factor(logo_feats$global.bm_label)
  
  colnames(accent_colors) <- paste("ac.", colnames(accent_colors), sep="")
  colnames(bin_fontclass) <- paste("binfont.",colnames(bin_fontclass), sep="")
  
  
  logo_feats[,select.subfeature("global","mmark")] <- 1*(logo_feats[,select.subfeature("global","mmark")] == "True")
  
  
  fontform <- select(logo_feats, font.form_sans, font.form_serif)
  formfactor <- remove.prefix(colnames(fontform), "_")[apply(fontform, 1, which.max)]
  formfactor[rowSums(fontform)==0] <- "nochars"
  
  
  
  
  
  y_mult <- cbind(select(logo_feats, global.domcolor, global.hull_type, global.bm_label), formfactor)
  y_mult$global.domcolor = factor(y_mult$global.domcolor, levels = sort(color_labels$desc[-1]))
  y_mult$global.hull_type = factor(y_mult$global.hull_type, levels = hull_labels)
  y_mult$global.bm_label = factor(y_mult$global.bm_label, levels = mark_orig_labels)
  y_mult$formfactor = factor(y_mult$formfactor, levels = c("nochars","sans","serif"))
  
  
  
  
  
  y_bin <- logo_feats[,select.feature("color") | select.subfeature("global","mmark") | select.subfeature("font","has")]
  y_bin <- y_bin[,-which(colnames(y_bin) == "color.white")]
  y_bin <- cbind(y_bin, 
                 select(logo_feats, "global.has_mark"), 
                 bin_fontclass,
                 accent_colors)
  y_bin[which(is.na(y_bin), arr.ind=TRUE)] <- 0
  y_bin[which(y_bin > 1, arr.ind=TRUE)] <- 1
  
  
  
  y_real <- logo_feats[,as.logical(select.feature("global") - 
                                     select.subfeature("global","mmark") - 
                                     (colnames(logo_feats) %in% c("global.domcolor", "global.hull_type", 
                                                                  "global.has_mark", "global.bm_label")))]
  
  y_real = select(y_real, -global.w, -global.h, -global.ar, -global.vert)
  y_real$global.h_sym[is.na(y_real$global.h_sym)] = mean(y_real$global.h_sym, na.rm = TRUE)
  y_real$global.v_sym[is.na(y_real$global.v_sym)] = mean(y_real$global.v_sym, na.rm = TRUE)
  
  colnames(y_real) = remove.prefix(colnames(y_real))
  
  y_bin$zero_sat = 1*(y_real$sd_sat == 0)
  
  
  
  y_bin$high_sat = 1*(y_real$mean_sat > high_sat_cut)
  y_bin$high_sat_sd = 1*(y_real$sd_sat > high_sat_sd_cut)
  y_bin$low_sat = 1*(y_real$mean_sat < low_sat_cut & y_real$mean_sat > 0)
  y_bin$low_sat_sd = 1*(y_real$sd_sat < low_sat_sd_cut & y_real$sd_sat > 0)
  
  y_real = select(y_real, -sd_sat, -mean_sat)
  y_count = select(y_real, ncolors, nmarks, nchars)
  y_real = select(y_real, -ncolors, -nmarks, -nchars)
  
  
  y_mult_py <- t(as.matrix(sapply(y_mult, function(x) as.numeric(x)-1)))
  rownames(y_mult_py) <- rownames(y_mult)
  colorkey <- apply(as.data.frame(y_mult), 2, function(x) t(t(levels(as.factor(x)))))
  
  y_real_high = 1*(y_real > y_real_high_cuts)
  y_real_low = 1*(y_real < y_real_low_cuts)
  
  colnames(y_real_low) = paste(colnames(y_real_low), ".", "low", sep="")
  colnames(y_real_high) = paste(colnames(y_real_high), ".", "high", sep="")
  
  y_bin_all = cbind(y_bin, y_real_low, y_real_high)
  
  y_bin_all$no_chars = 1*(y_count$nchars == 0)
  y_bin_all$many_chars = 1*(y_count$nchars > quantile(y_count$nchars, 0.75))
  y_bin_all$many_marks = 1*(y_count$nmarks > 2)
  
  y_mult2 = data.frame(y_mult_py)
  y_mult2$ncolors = ifelse(y_count$ncolors == 2, 1, ifelse(y_count$ncolors == 3, 2, ifelse(y_count$ncolors == 4, 3, 4))) - 1
  
  list(mult_feats = y_mult2, bin_feats = y_bin_all)
}

# This just wraps the above two functions:
create.logo.features = function(name, colors, hulls, fonts, global_feats, marks, mark_feats){
  merged_feats = create.merged.feats(name, colors, hulls, fonts, global_feats, marks, mark_feats)
  processed_feats = process.features(name, merged_feats)
  
  processed_feats
}


